#fonction permettant le calcul du prix
simu_rendement <- function(r,T,rho,vol1,vol2)
{
  N = 100000
  U = matrix(rnorm(N*2),N,2)
  C = matrix(c(1,rho,rho,1),2,2)
  S = chol(C)
  V = matrix(0,2,N)
  for (i in seq(1,N)) {V[,i] = t(S)%*%U[i,]}
  V = t(V)
  #cor(V)
  S1 = exp(r*T+vol1*sqrt(T)*V[,1])
  S2 = exp(r*T+vol2*sqrt(T)*V[,2])
  M = apply(cbind(S2-S1,matrix(0,100000,1)),1,'max')
  return(mean(M)*exp(-r*T))
}

# application permettant le passage du plan d'expériences au plan d'expérimentation
# les bornes utilisées sont celles de l'exercice
# l'ordre des variables correspond à r, T, rho, sigma1, sigma2 
rescale <- function(X)
{
  bornes= matrix(c(0.02,0.07,0.5,5,0.2,0.9,0.1,0.4,0.1,0.4),2,5)
  X2 = X
  for (j in seq(1,ncol(X)))
   {
    a = bornes[1,j]
    b = bornes[2,j]
    X2[,j] = (a+b)/2+X[,j]*(b-a)/2
  }  
  return(X2)
}

