library("mice")
library("VIM")

all_df <- read.csv("C:/Users/Chi/Desktop/house price predict/all_df_encoded.csv")

all_df[,'Id'] -> idrm

all_df$'Id' <- NULL

for(i in names(all_df)){
  
  if ( length(unique(all_df[,i])) <= 16) {
    all_df[,i] <- as.factor(all_df[,i])
  }
}

micemod <-  mice(all_df,m=20,maxit =20, method='rf')

all_df_filled <- complete(micemod)

all_df_filled[,'Id'] <- id

write.csv(all_df_filled,"C:/Users/Chi/Desktop/house price predict/all_df_filled.csv", row.names=FALSE)
