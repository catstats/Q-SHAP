library(qshapr)

# 1 indexed unlike [9, :9] in python
print(qshapr::r_store_complex_v_invc(20)[10, 1:9])

v1 <- c(3, 6 - 1i, 5 + 1i, 6 + 1i)
v2 <- c(5, 4 - 1i, 6 + 1i, 4 + 1i)
v3 <- c(3, 6 - 1i, 5 + 1i)
v4 <- c(5, 4 - 1i, 6 + 1i)
print(qshapr::r_complex_dot_v2(v3, v4, length(v1)))

v5 <- c(3, 6 - 1i, 5 + 1i, 5 - 1i, 6 + 1i)
v6 <- c(5, 4 - 1i, 6 + 1i, 6 - 1i, 4 + 1i)
v7 <- c(3, 6 - 1i, 5 + 1i)
v8 <- c(5, 4 - 1i, 6 + 1i)
print(qshapr::r_complex_dot_v2(v7, v8, length(v5)))