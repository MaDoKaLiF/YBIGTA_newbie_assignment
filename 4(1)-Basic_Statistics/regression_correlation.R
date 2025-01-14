rm(list=ls())

### Advertising.csv를 불러와 데이터 로드하기!
advertising_data <- read.csv("Advertising.csv")



### Multiple Linear Regression을 수행해봅시다!
# 'Sales'를 종속 변수로, 'TV' 'Radio', 'Newspaper'를 독립 변수로 설정
mlr_model <- lm(sales ~ TV + radio + newspaper, data = advertising_data)
summary(mlr_model)  


### Correlation Matrix를 만들어 출력해주세요!
correlation_matrix <- cor(advertising_data[, c("TV", "radio", "newspaper", "sales")])
print(correlation_matrix)
