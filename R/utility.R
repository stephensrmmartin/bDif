summary_chains <- function(bDif,chains=bDif@chain.max,pars,...){
	monitor(extract(bDif,pars=pars,permuted=FALSE)[,chains,,drop=FALSE],warmup=0,print=FALSE,...)
}
invLogit <- function(logits){
	exp(logits)/(1 + exp(logits))
}

invLogit.lambda <- function(lambdaVector){
	pis <- invLogit(lambdaVector)
	piMatrix <- matrix(c(1-pis,pis),ncol=2)
	piMatrix
}

softmax <- function(logitVector){
	exp(logitVector)/sum(exp(logitVector))
}
softmax.lambdaMatrix <- function(lambdaMatrix){
	piMatrix <- t(apply(lambdaMatrix,MARGIN = 1,softmax))
	piMatrix
}

getModalCluster <- function(piMatrix){
	apply(piMatrix,1,which.max)
}