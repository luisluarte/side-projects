
x <- array(rnorm(1000 * 4 * 10), dim = c(1000, 4, 10))
v <- apply(x, 3, var)
cat('Class:', class(v), '\n')
cat('Dim:', dim(v), '\n')

