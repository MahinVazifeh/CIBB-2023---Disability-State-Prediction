
rm(list = ls())
# 1.	Import original data
# 2.	Test and Train separate (20,80)
# 3.	Impute with 5 methods for only train data
# 4.  Cross validation (train (train o validation))
# 5.	prediction with 4 methods (lightgboost, knn, svm, random forest) 
# -	5 * 4  = 20 possible status
# 6.	The best R 2 score 
# 7.	Test original data


# Import libraries
library(dplyr)
library(mice)
library(jomo)
library(mitml)
library(bruceR)
library(caret)
library(mitml)
library(imputeTS)
library(splitTools)
library(groupdata2)
library(e1071)
library(randomForest)
library(lightgbm)

# Define functions
RSQUARE <- function(y_actual,y_predict){
  cor(y_actual,y_predict)^2}

Jomo_imputation <- function(incomplete_data){
  
  Y <- incomplete_data %>% select(Pyramidal:Deambulation)
  no_levels <- function(x) length(unique(na.omit(x)))
  Y.numcat <- apply(Y,2,no_levels)
  X <- incomplete_data %>% select(EDSS:Age)
  clus <- incomplete_data %>% select(Patient_ID)
  
  nburn <- 10L
  nbetween <- 10L
  nimp <- 2L
  
  jomo_imputed <- jomo1rancat(Y = Y, Y.numcat= Y.numcat, X = X, clus = clus, nburn = nburn, 
                              nbetween = nbetween, nimp = nimp)
  # select imputation number
  i = 1
  imputed_data <- jomo_imputed %>% filter(Imputation==i)
  imputed_data_subset <- imputed_data %>% 
    select(Patient_ID = clus, Pyramidal:Age) %>% 
    mutate_if(is.factor,as.numeric) %>% as.data.frame()
  return(imputed_data_subset)}


# Imputation with other methods
five_imputation <- function(incomplete_data,method = "all"){
  
  if(method=="all"){
    
    linear_imputed <- na_interpolation(incomplete_data)
    weighted_imputed <- na_ma(incomplete_data, k = 4, weighting = "exponential", maxgap = Inf)
    
    imp_1 <- mice(incomplete_data, m=5, maxit=30, meth="cart", seed=789)
    cart_imputed <- complete(imp_1)
    
    imp_2 <- mice(incomplete_data, m=5, maxit=30, meth="rf", seed=789)
    rf_imputed <- complete(imp_2)
    
    jomo_imputed <- Jomo_imputation(incomplete_data)
    return(list(linear_imputed, weighted_imputed, cart_imputed, rf_imputed,jomo_imputed))
    
  }else if(method =="Linear"){
    linear_imputed <- na_interpolation(incomplete_data)
    return(linear_imputed)
    
  }else if(method =="EWMA"){
    weighted_imputed <- na_ma(incomplete_data, k = 4, weighting = "exponential", maxgap = Inf)
    return(weighted_imputed)
    
  }else if(method =="MI_CART"){
    imp_1 <- mice(incomplete_data, m=5, maxit=50, meth="cart", seed=789)
    cart_imputed <- complete(imp_1)
    return(cart_imputed)
    
  }else if(method =="MI_RF"){
    imp_2 <- mice(incomplete_data, m=5, maxit=50, meth="rf", seed=789)
    rf_imputed <- complete(imp_2)
    return(rf_imputed)
    
  }else if(method =="JOMO"){
    jomo_imputed <- Jomo_imputation(incomplete_data)
    return(jomo_imputed)
  }
}

four_prediction <- function(x_train,y_train,x_valid,y_valid,neighbor,ntree,method="all"){
  
  if(method=="all"){
    
    model_knn <- knnreg(x_train, y_train, k = neighbor)
    y_pred <- predict(model_knn,newdata = x_valid)
    r2_knn <- RSQUARE(y_valid,y_pred)
    
    model_svm <- svm(x_train, y_train, kernel = "radial")
    y_pred <- predict(model_svm,newdata = x_valid)
    r2_svm <- RSQUARE(y_valid,y_pred)
    
    model_rf <- randomForest(x_train, y_train, ntree = ntree)
    y_pred <- predict(model_rf,newdata = x_valid)
    r2_rf <- RSQUARE(y_valid,y_pred)
    
    dtrain <- lgb.Dataset(data=as.matrix(x_train), label=y_train)
    dtest <- lgb.Dataset.create.valid(dtrain, x_valid, label = y_valid)
    model_lgb <- lightgbm(boosting_type = 'gbdt', 
                          objective = "regression", 
                          metric = 'mae', 
                          dtrain, 
                          nrounds = 500,
                          verbose = -1)
    y_pred <- predict(model_lgb, as.matrix(x_valid))
    r2_lg <- RSQUARE(y_valid,y_pred)
    
    r2_all <- c(KNN = r2_knn, SVM = r2_svm, RF = r2_rf, LG = r2_lg)
    return(r2_all)
    
  }else if(method=="KNN"){
    model_knn <- knnreg(x_train, y_train, k = neighbor)
    y_pred <- predict(model_knn,newdata = x_valid)
    r2_knn <- RSQUARE(y_valid,y_pred)
    return(r2_knn)
    
  }else if(method=="SVM"){
    model_svm <- svm(x_train, y_train, kernel = "radial")
    y_pred <- predict(model_svm,newdata = x_valid)
    r2_svm <- RSQUARE(y_valid,y_pred)
    return(r2_svm)
    
  }else if(method=="RF"){
    model_rf <- randomForest(x_train, y_train, ntree = ntree)
    y_pred <- predict(model_rf,newdata = x_valid)
    r2_rf <- RSQUARE(y_valid,y_pred)
    return(r2_rf)
    
  }else if(method=="LG"){
    dtrain <- lgb.Dataset(data=as.matrix(x_train), label=y_train)
    dtest <- lgb.Dataset.create.valid(dtrain, x_valid, label = y_valid)
    model_lgb <- lightgbm(boosting_type = 'gbdt', 
                          objective = "regression", 
                          metric = 'mae', 
                          dtrain, 
                          nrounds = 500,
                          verbose = -1)
    y_pred <- predict(model_lgb, as.matrix(x_valid))
    r2_lg <- RSQUARE(y_valid,y_pred)
    return(r2_lg)
  }
}


# Step 1: Import raw csv data
Original_data <- read.csv(".../Updated_MS_14000.csv", header = TRUE)

# Step 2: Train and Test Seperate considering Patient id
raw_data <- Original_data %>% 
  select(Patient_ID,Pyramidal:Deambulation,EDSS = EDSS_score_assessed_by_clinician,Sex:Age)

# Step 3: Train and Test Seperate considering Patient id
set.seed(123)
ids <- splitTools::partition(raw_data$Patient_ID, p = c(train = 0.8, test = 0.2), type = "grouped")
tr_data <- raw_data[ids$train,]
te_data <- raw_data[ids$test,]

imputation_result <- five_imputation(tr_data)
save(imputation_result,file= "imputation_result.RData")
load("imputation_result.RData")

my_neighbor <- c(3,5)
my_ntree <- c(100,200,300)
imp_names <- c("Linear","EWMA","MI_CART","MI_RF","JOMO")
result_jomo <- c()
i <- 5
# Loop through imputation methods
for (i in seq_along(imputation_result)) {
  selected_data <- imputation_result[[i]]
  k = 10 # The number of folds
  selected_data$Patient_ID <- as.factor(selected_data$Patient_ID)
  #Loop through neighborhood values
  for (n in seq_along(my_neighbor)){
    #Loop through ntree values
    for (nt in seq_along(my_ntree)) {
      tr_data_folds <- groupdata2::fold(selected_data,k = k, id_col ="Patient_ID")
      # Cross Validation with Ten Folds on the selected data
      for (j in 1:k){
        
        CV_train <- tr_data_folds %>% filter(.folds != j)
        CV_valid <- tr_data_folds %>% filter(.folds == j)
        y_train <- CV_train %>% ungroup() %>% pull(EDSS)
        x_train <- CV_train %>% ungroup() %>% select(-EDSS,-.folds ,-Patient_ID)
        y_valid <- CV_valid %>% ungroup() %>% pull(EDSS)
        x_valid <- CV_valid %>% ungroup() %>% select(-EDSS,-.folds ,-Patient_ID)
        
        fit <- four_prediction(x_train,
                               y_train,x_valid,y_valid,
                               neighbor = my_neighbor[n],
                               ntree = my_ntree[nt])
        
        method_names <- names(fit)
        r2_values <- unname(fit)
        
        res <- data.frame(imputation=imp_names[i],
                          neighbor = my_neighbor[n],
                          ntree = my_ntree[nt],
                          fold = j,
                          method = method_names,
                          R2=r2_values)
        
        result_jomo <- rbind.data.frame(result_jomo,res)
        cat(paste("Imputation =",imp_names[i],
                  ", neighbor =",my_neighbor[n],
                  ", ntree =",my_ntree[nt],
                  ", fold =",j,
                  ", method =",method_names
        ),sep = "\n")
      }
    }
  }
}
result <- rbind.data.frame(result,result_jomo)
load(result)
View(result)
save(result,file= "result_full.RData")
result <- get(load("result_full.RData"))

result_maxfold <- result %>% 
  group_by(imputation,neighbor,ntree,method) %>% 
  summarise(R2=mean(R2))
write.csv(result_maxfold,"result_maxfold.csv",row.names = FALSE)
View(result_maxfold)
result_1 <- get(load("result_maxfold"))
result_all <- result_maxfold %>%
  group_by(imputation,method,neighbor) %>% 
  slice_max(R2) %>%
  ungroup() %>% 
  group_by(imputation,method) %>% 
  slice_max(neighbor) %>% 
  ungroup() %>% 
  group_by(imputation) %>%
  mutate(max=ifelse(R2==max(R2),"*",NA)) %>% 
  relocate(method,.after = imputation)
write.csv(result_all,"result_all.csv",row.names = FALSE)

View(result_all)

result_report <- result_maxfold %>%
  group_by(imputation,method,neighbor) %>% 
  slice_max(R2) %>%
  ungroup() %>% 
  group_by(imputation,method) %>% 
  slice_max(neighbor) %>% 
  select(-neighbor,-ntree) %>% 
  ungroup() %>% 
  group_by(imputation) %>%
  mutate(max=ifelse(R2==max(R2),"*",NA)) %>% 
  relocate(method,.after = imputation)
write.csv(result_report,"Table3.csv",row.names = FALSE)
getwd()

# Select the best rows for test data
result_final_0 <- result_all %>%
  filter(max=="*") %>% 
  select(-max)
write.csv(result_final_0,"result_final_0.csv",row.names = FALSE)

# Predict on the original test data based on the optimal scenario
result_final <- result_final_0 %>% 
  mutate(R2_test=NA)
for (sc in 1:nrow(result_final)){
  
  imp_method <- result_final$imputation[sc]
  pre_method <- result_final$method[sc]
  
  neighbor <- result_final$neighbor[sc]
  ntree <- result_final$ntree[sc]
  
  imputed_test <- five_imputation(te_data, method =imp_method)
  imputed_train <- five_imputation(tr_data, method =imp_method)
  
  y_train <- imputed_train %>% pull(EDSS)
  x_train <- imputed_train %>% select(-EDSS,-Patient_ID)
  
  y_test <- imputed_test %>% pull(EDSS)
  x_test <- imputed_test %>% select(-EDSS,-Patient_ID)
  
  model <- four_prediction(x_train,
                           y_train,
                           x_test,
                           y_test,
                           neighbor,
                           ntree,
                           method=pre_method)
  
  cat(paste("neighbor =",neighbor,
            "and ntree =",ntree,
            "and imputation =",imp_method,
            "and method =",pre_method,
            "is ",model
            ),sep = "\n")
  # save the result(the final R2)
  result_final$R2_test[sc] <- model
  
}
View(result_final)
write.csv(result_final,"result_final.csv",row.names = FALSE)







