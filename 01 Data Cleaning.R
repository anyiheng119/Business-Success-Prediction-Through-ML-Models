
setwd("E:/UCLA Econ Course/425 Machine Learning I/Final Project")

# Load libraries
library(tidyverse)
library(plotly)
library(ggthemes)
library(lubridate)

# Import datasets
company = data.table::fread("companies.csv", na.strings = "")
investments = data.table::fread("investments.csv", na.strings = "")
acquisitions = data.table::fread("acquisitions.csv", na.strings = "")
exchange = data.table::fread("exchange rate gold.csv",header = T, na.strings = "")

# Checks size of datasets
print(object.size(company), units = 'Mb')
print(object.size(investments), units = 'Mb')
print(object.size(acquisitions), units = 'Mb')

# Get a quick overview of the data structure
glimpse(company)
glimpse(investments)
glimpse(acquisitions)

# Remove duplicates (for company names)
nrow(company)                                       # 66368
nrow(company %>% distinct(name, .keep_all = TRUE))  # 66103
company = company %>% distinct(name, .keep_all = TRUE)

# Keep companies founded only between 2004 and 2015
company = filter(company,year(founded_at) <= 2015)
company = filter(company,year(founded_at) >= 2004)
company = company[,c(-8,-9,-10)] # drop state, region and city 

# Investment duration
b = company$last_funding_at 
a = company$first_funding_at
company$invest_duration = as.double(difftime(b,a,units="weeks"))  # unit is weeks

# clean category list
index = grep("[|]",company$category_list)  # row index which include "|" in category list
company[index,]$category_list = gsub("\\|.*$", "", company[index,]$category_list)  # only keep the first category before "|"
table(company$category_list)

# NAs visualization
company$funding_total_usd = gsub("-", NA, company$funding_total_usd)
missing_data = company %>% summarise_all(funs(sum(is.na(.))/n()))
missing_data = gather(missing_data, key = "variables", value = "percent_missing")
ggplot(missing_data, aes(x = reorder(variables, percent_missing), y = percent_missing)) +
  geom_bar(stat = "identity", fill = "darkgreen", aes(color = I('white')), size = 0.3)+
  xlab('variables')+
  coord_flip()+ 
  theme_bw()

# clean company data
company.clean = na.omit(company)
glimpse(company.clean)

# Compute invest deals by companies
investor_group = group_by(investments, company_permalink)
invest_deals = summarise(investor_group, deals = n())
colnames(invest_deals)[1] = "permalink"

# Clean exchange rates
exchange = na.omit(exchange)
exchange$ex_sd = apply(exchange[,c(3:19)], 1, sd)	# compute sd by row

# Join datasets
company.clean = left_join(company.clean,invest_deals,by="permalink")
company.clean = left_join(company.clean,exchange[,c(1,20)],by="country_code")
company.clean = na.omit(company.clean)
company.clean = company.clean[,c(-2,-3)] # drop name and webpage
data = company.clean
table(year(data$founded_at))

################################################################################

# Exchange rate variation over 5 years since the company founded
for (i in 1:nrow(data)){

  df = subset(exchange,country_code == as.character(data[i, 5])) # matcah country code
  s = year(data$founded_at[i])-2001                        # 
  data$ex_sd_5years[i] = apply(df[,s:(s+5-1)], 1, sd)      # sd of Exchange rate in 5 years since the company founded
  data$ex_mean_5years[i] = apply(df[,s:(s+5-1)], 1, mean)  # mean of Exchange rate in 5 years since the company founded
}


# Google trend data
google_IPO = read.csv("Google trend IPO.csv")
google_MA = read.csv("Google trend M&A.csv")
colnames(google_MA)[1] = "date"

glimpse(google_IPO)
glimpse(google_MA)


# IPO features
for (i in 1:nrow(data)){
  df = google_IPO[ ,which(colnames(google_IPO) %in% as.character(data[i, 5]))] # match the country code
  s = year(data$founded_at[i])-2003
  start = (s-1)*12 + 1
  over = start + 12*5 - 1
  data$IPO_mean_5years[i] = mean(df[start:over])                   # mean of IPO search score over 5 years since the company founded
  data$IPO_growth_5years[i] = (df[over] - df[start]) / df[start]   # growth rate of IPO search score over 5 years since the company founded
}


# M&A features
for (i in 1:nrow(data)){
  df = google_MA[ ,which(colnames(google_MA) %in% as.character(data[i, 5]))] # match the country code
  s = year(data$founded_at[i])-2003
  start = (s-1)*12 + 1
  over = start + 12*5 - 1
  data$MA_mean_5years[i] = mean(df[start:over])                   # mean of M&A search score over 5 years since the company founded
  data$MA_growth_5years[i] = (df[over] - df[start]) / df[start]   # growth rate of M&A search score over 5 years since the company founded
}

# Remove growth rate NAs
data$IPO_growth_5years = gsub("Inf", NA, data$IPO_growth_5years)
data$IPO_growth_5years = gsub("-Inf", NA, data$IPO_growth_5years)
data$MA_growth_5years = gsub("Inf", NA, data$MA_growth_5years)
data$MA_growth_5years = gsub("-Inf", NA, data$MA_growth_5years)
data = na.omit(data)

# Correct data types
str(data)
data$funding_total_usd = as.numeric(data$funding_total_usd)
data$IPO_growth_5years = as.numeric(data$IPO_growth_5years)
data$MA_growth_5years = as.numeric(data$MA_growth_5years)


# Label the data
table(company$status)
data$label = ifelse(data$status == "acquired" | data$status == "ipo",1,0)
data$label = as.factor(data$label)
df = data[,c(-1,-4,-6,-7,-8,-9)]
df = na.omit(df)
str(df)

# output clean data
write.csv(df, "clean_data.csv")


# one-hot for character features
length(table(df$country_code))  # 59
length(table(df$category_list)) # 608, too large

data.hot = model.matrix(label~country_code-1, df) # one-hot coding for country code
data.hot = as.data.frame(data.hot)
data.hot = cbind(data.hot, df[,c(-1,-3)])
str(data.hot)

# output clean one-hot data
write.csv(data.hot,"clean_data_hot.csv")

