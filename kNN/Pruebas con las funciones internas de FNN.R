
datos <- query <- cbind(c(3, 5, 1, 8, 3), c(3, 6, 7, 4, 1))
colnames(datos) <- c("x1", "x2")
datos

get.knn(datos, k=3)
get.knnx(datos, query, k=5)
get.knnx(datos, query, k=5, algo="kd_tree")

th<- runif(10, min=0, max=2*pi)
data2<-  cbind(cos(th), sin(th))
get.knn(data2, k=5, algo="CR")

plot(datos)
dist(datos)
