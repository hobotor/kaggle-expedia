#задача   - expedia.com
#источник - kaggle.com
#данные   - https://www.kaggle.com/c/expedia-hotel-recommendations/data

#model 14 версия 2.1
#валидация:
# пытаемся прогнозировать 2014-12 месяц
# обучающая таблица с 2013-01 по  2014-11 месяц

#показатель            - MAP@5
#достигнутый результат - 0.41

##################################################################################################################

#загружаем необходимые библиотеки
#library(bizdays)
#library(matrixStats)
#library(moments)

library(zoo)
library(Metrics)
library(data.table)
library(Matrix)
library(xgboost)

##########################
#функция подсчета исторических данных
prep_history<- function(booking=1,fact="user_id",letter="u",lst_date=date_lst,saved=T,ret=F){
  
  Max_i=length(lst_date)
  pb <- winProgressBar(title=paste0("Progress bar ",fact), label="0% done", min=0, max=100, initial=0)
  
  if(!exists("expedia_train")) {load("expedia_train.rd")
    
    expedia_train[,hotel_cluster:=factor(hotel_cluster,levels=as.character(0:99))]
  }
  fact.lst=list()
  et=expedia_train[is_booking==booking,mget(c(fact,"cnt","hotel_cluster","yearmonth_srch"))]
  i=0
  for(dl in lst_date){
    fact.tab=et[yearmonth_srch<dl,.(N=sum(cnt,na.rm=T)),by=mget(c(fact,"hotel_cluster"))]
    fact.tab[,rs:=sum(N,na.rm=T),by=mget(fact)]
    fact.tab[,N:=round(N/rs,2)]
    fact.tab=dcast(fact.tab,as.formula(paste0(fact,"~hotel_cluster")),value.var = "N",fun.aggregate = sum,na.rm=T,fill=0,drop=c(T,F))
    setnames(fact.tab,paste0(0:99),paste0(letter,booking,"hc",0:99))
    fact.tab[,yearmonth_srch:=as.integer(dl)]
    if (saved) save(fact.tab,file=paste0("final/",fact,"_hc",booking,"_train_",dl,".rd"))
    if (ret) fact.lst[[dl]]=fact.tab
    
    i=i+1
    info <- sprintf("%d%% done", round((i/Max_i)*100))
    setWinProgressBar(pb, i/Max_i*100, label=paste0(info,"  ",dl))
    
  }
  close(pb)
  return(fact.lst)
}
###############################
#функция подключения исторических данных к обучающей таблице

prep_tab<-function(dt=tt1,lst_fact=fact_lst,date_lim=dl){
  for(fact in lst_fact){
    load(paste0("final/",fact,"_hc",1,"_train_",date_lim,".rd"))
    fact.tab[,yearmonth_srch:=as.integer(yearmonth_srch)]
    dt=merge(dt,fact.tab,by=c("yearmonth_srch",fact),all.x=T,sort=F)
    load(paste0("final/",fact,"_hc",0,"_train_",date_lim,".rd"))
    fact.tab[,yearmonth_srch:=as.integer(yearmonth_srch)]
    dt=merge(dt,fact.tab,by=c("yearmonth_srch",fact),all.x=T,sort=F)
  }
  return(dt)
}

########################################################
# функция расчета оценки для xgboost


if(!exists("evalerror")) {
  
  evalerror <- function(preds, dtrain) {
    #browser()
    labels <- getinfo(dtrain, "label")
    pred1<-data.table(matrix(preds,nrow=100,ncol=length(labels)))
    err<-mapk(5,labels,pred1[,lapply(.SD,function(x) as.character(0:99)[order(x,decreasing=T)])][1:5,])
    return(list(metric = "map@5val", value = err))
  }
}
########################################################
#берем основные данные
#оставляем отобранные в результате доп анализа координаты
fields_lst=c("hotel_cluster","user_id","srch_destination_id","user_location_city","hotel_market","cnt","is_booking","date_time")
#,cnt=NULL)
expedia_train <- fread('D:/RData/expedia/input/train.csv', header=TRUE,select=fields_lst)


#месяц поиска отеля as.integer(gsub("-","","2014-08"))
#system.time(expedia_train[,yearmonth_srch:=as.integer(gsub("-","",substr(date_time,1,7)))])out of memmory

#форматируем date_time
expedia_train[,date_time:= as.POSIXct(date_time,origin="1970-01-01")]
expedia_train[,yearmonth_srch:=(as.POSIXlt(date_time)$year+1900L)*100L+as.POSIXlt(date_time)$mon+1L]
#expedia_train[,date_time:=NULL]
#save(expedia_train,file="final/expedia_train.rd")
#load("final/expedia_train.rd")
date_lst=c(201412L:201401L,201312L:201311L)

#готовим исторические данные
prep_history()
prep_history(booking = 0)

prep_history(booking = 0,fact = "srch_destination_id",letter="sd")
prep_history(booking = 1,fact = "srch_destination_id",letter="sd")

prep_history(booking = 0,fact = "user_location_city",letter="ulc")
prep_history(booking = 1,fact = "user_location_city",letter="ulc")

prep_history(booking = 0,fact = "hotel_market",letter="hm")
prep_history(booking = 1,fact = "hotel_market",letter="hm")

#делим на обучающую и тестовую часть
#expedia_train[,spl:=ifelse(yearmonth_srch<201411,0L,1L)]
#et.lst=split(expedia_train,by="spl",keep.by = F)
#в тестовой части оставляем только бронирование

#меняем date_time на yearmonth_srch
fields_lst=c("hotel_cluster","user_id","srch_destination_id","user_location_city","hotel_market","yearmonth_srch")

tt=expedia_train[is_booking==1&yearmonth_srch==201412,mget(fields_lst)]
tr=expedia_train[is_booking==1&yearmonth_srch<201412&yearmonth_srch>201310,mget(fields_lst)]
#rm(expedia_train)  #=expedia_train[yearmonth_srch<201411]

gc()

set.seed(4)
size_sample=0.02
yearmonth_train.lst=split(tr[,.(id=1:.N,yearmonth_srch)],by="yearmonth_srch")#hotel_cluster")
index_train <- as.integer(unlist(sapply(yearmonth_train.lst, function(x) x[,id][sample.int(nrow(x), max(1,size_sample*nrow(x)), FALSE)])))
tr1=tr[index_train]

#для валидации оставим только бывших ранее клиентов
for (nm in c("user_id")){     
  tt=tt[get(nm)%in%tr1[,unique(get(nm))]]
}
set.seed(2)
index_valid=sample.int(nrow(tt),500)

#load("tt1.rd")

tt1=tt[index_valid]
tr1=tr[index_train]

dl=date_lst[1]
fact_lst=c("user_id","srch_destination_id","user_location_city","hotel_market")
prep_tab<-function(dt=tt1,lst_fact=fact_lst,date_lim=dl)
for(fact in fact_lst){
  load(paste0("final/",fact,"_hc",1,"_train_",dl,".rd"))
  fact.tab[,yearmonth_srch:=as.integer(yearmonth_srch)]
  tt1=merge(tt1,fact.tab,by=c("yearmonth_srch",fact),all.x=T,sort=F)
  load(paste0("final/",fact,"_hc",0,"_train_",dl,".rd"))
  fact.tab[,yearmonth_srch:=as.integer(yearmonth_srch)]
  tt1=merge(tt1,fact.tab,by=c("yearmonth_srch",fact),all.x=T,sort=F)
}
tt1.lst=tt1

tr1.lst=split(tr1,by="yearmonth_srch")
Max_i=length(date_lst)
pb <- winProgressBar(title="Progress bar", label="0% done", min=0, max=100, initial=0)
i=0

for( dl in date_lst[-1]){
  dl=as.character(dl)
  for(fact in fact_lst){
    load(paste0("final/",fact,"_hc",1,"_train_",dl,".rd"))
    fact.tab[,yearmonth_srch:=as.integer(yearmonth_srch)]
    tr1.lst[[dl]]=merge(tr1.lst[[dl]],fact.tab,by=c("yearmonth_srch",fact),all.x=T,sort=F)
    load(paste0("final/",fact,"_hc",0,"_train_",dl,".rd"))
    fact.tab[,yearmonth_srch:=as.integer(yearmonth_srch)]
    tr1.lst[[dl]]=merge(tr1.lst[[dl]],fact.tab,by=c("yearmonth_srch",fact),all.x=T,sort=F)
  }
  
  
  i=i+1
  info <- sprintf("%d%% done", round((i/Max_i)*100))
  setWinProgressBar(pb, i/Max_i*100, label=paste0(info,"  ",dl))
  
}
close(pb)



tr1=rbindlist(tr1.lst)
tt1=copy(tt1.lst)
#rm(tr1.lst)

tt1[,(names(tt1)[-c(1,6)]):=lapply(.SD, na.fill,fill=0),.SDcols=-c(1,6)]
#корректируем там, где вообще небыло значений и поэтому деление на 0
tr1[,(names(tt1)[-c(1,6)]):=lapply(.SD, na.fill,fill=0),.SDcols=-c(1,6)]

stopifnot(!anyNA(tr1))
stopifnot(!anyNA(tt1))


fields_lst=names(tt1.lst)[c(6,2,3,4,11:410)]  

tt1=tt1[                     ,mget(fields_lst)]

tt2=tr1[yearmonth_srch==11311,mget(fields_lst)]
tt3=tr1[yearmonth_srch==11410,mget(fields_lst)]
tr1=tr1[                     ,mget(fields_lst)]

############################################################################ 

trm=sparse.model.matrix(hotel_cluster~-1+.,data=tr1)

ttm1=sparse.model.matrix(hotel_cluster~-1+.,data=tt1)
ttm2=sparse.model.matrix(hotel_cluster~-1+.,data=tt2)
ttm3=sparse.model.matrix(hotel_cluster~-1+.,data=tt3)

gc()

dtrain <- xgb.DMatrix(data=trm,label=tr1[,hotel_cluster])
dvalid1 <- xgb.DMatrix(data=ttm1,label=tt1[,hotel_cluster])
#dvalid2 <- xgb.DMatrix(data=ttm2,label=tt2[,hotel_cluster])
#dvalid3 <- xgb.DMatrix(data=ttm3,label=tt3[,hotel_cluster])
watchlist <- list(train=dtrain,valid1=dvalid1)


param <- list(  objective           = "multi:softprob",#binary:logistic",#logitraw", 
                booster             = "gbtree",#"gblinear",#
                eval_metric         = evalerror,#"merror",#mlogloss",#map@5",#auc",#map@5",#logloss",
                num_class           = 100,
                #xgb.model           = bst_pred,
                #scale_pos_weight    = sumwneg / sumwpos,
                eta                 = 0.1,
                max_depth           = 6#,
                #subsample           = 0.5 #,
                #colsample_bytree    = 0.3
)
set.seed(2)
system.time(
  bst <- xgb.train(   params                   = param, 
                      data                     = dtrain, 
                      nrounds                  = 50, 
                      save_period              = 0,
                      #save_name                = (paste0(fd,".mdl")),
                      verbose                  = 1,
                      watchlist                = watchlist,
                      maximize                 = T,
                      print_every_n            = 1,
                      early_stopping_rounds    = 10
  )
)
 

tt=prep_tab(dt=tt,lst_fact=fact_lst,date_lim=201412)
tt[,(names(tt)[-c(1,6)]):=lapply(.SD, na.fill,fill=0),.SDcols=-c(1,6)]

stopifnot(!anyNA(tt))
fields_lst=names(tt1.lst)[c(6,2,3,4,11:410)]
tt=tt[                     ,mget(fields_lst)]
ttm=sparse.model.matrix(hotel_cluster~-1+.,data=tt)
dtest <- xgb.DMatrix(data=ttm,label=tt[,hotel_cluster])
preds=predict(bst,dtest)
evalerror(preds,dtest)
#$metric
#[1] "map@5val"
#
#$value
#[1] 0.3392486
evalerror(predict(bst,dvalid1),dvalid1)
#$metric
#[1] "map@5val"
#
#$value
#[1] 0.4099667

#справочно
#какова доля попаданий выбранного кластера в предлагаемую 10-ку для просмотра
k=10
preds.tab=data.table(id=rep(1:(length(preds)/100L),each=100),hotel_cluster=0:99,preds)
p1=preds.tab[,.(prom_hotel_cluster=paste(hotel_cluster[order(preds,decreasing = T)][1:k],collapse=",")),by=id]
p1[,hotel_cluster:=tt[,hotel_cluster]]
p1[,y_no:=hotel_cluster %in% unlist(strsplit(prom_hotel_cluster,",")),by=1:nrow(p1)]
p1[,sum(y_no)/nrow(p1)]
#[1] 0.7568298
