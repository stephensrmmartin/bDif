# bDif class functions ------------------ 
# Class creation
bDif <- setClass('bDif',contains = 'stanfit',slots = c('data','K','model.type','chain.max'))

#' 
bDifFit <- function(data,measurementModel,K,order,covariateModel,model.type,method='mcmc',...){
	responseMatrix <- model.matrix(measurementModel,data)[,-1]
	responseMissing <- !complete.cases(responseMatrix)
	responseMatrix <- responseMatrix[!responseMissing,]
	warning('Removing ',sum(responseMissing),' cases missing response data.')
	
	covariateMatrix <- model.matrix(covariateModel,data)
	covariateMissing <- !complete.cases(covariateMatrix)
	covariateMatrix <- covariateMatrix[!covariateMissing,,drop=FALSE]
	warning('Removing ',sum(covariateMissing),' cases missing covariate data.')
	
	N <- nrow(responseMatrix)
	L <- ncol(covariateMatrix)
	J <- ncol(responseMatrix)
	
	stan_data <- list(y=responseMatrix,covariates=covariateMatrix,K=K,jOrder=order,L=L,N=N,J=J)
	if(model.type == '2PL'){
		#pars <- c('alpha','diff','delta_logit','pi_logit','theta','betas_logit','log_lik')
		pars <- c('alpha','diff','delta_logit','pi_logit','theta','betas_logit','alpha_nondif','diff_nondif')
		if(K == 2){
			model_file <- 'Models/difSimplex3.stan'
		} else {
			model_file <- 'Models/difSimplex2.stan'
		}
	} else {
		stop('Only 2PL is currently supported.')
	}
	if(method=='mcmc'){
		stanOut <- stan(file=model_file,data = stan_data,pars=pars,...)
	} else if(method=='vb'){
		stanOut <- vb(stan_model(file=model_file),data = stan_data,pars=pars,algorithm='fullrank')
	}
		
	if(model.type == '2PL'){
		bDifOut <- bDif(stanOut,K=K,model.type=model.type,data=responseMatrix,chain.max=which.max(get_posterior_mean(stanOut,pars='lp__')))
	} 
	
	bDifOut
}

#Cluster membership
setGeneric('clusters',function(object,...){standardGeneric('clusters')})
setMethod('clusters',signature = c(object='bDif'),function(object,chains=object@chain.max,modal=TRUE){
	lambdas <- summary_chains(object,chains=chains,pars=c('pi_logit'))[,1]
	if(object@K == 2){
		piMatrix <- invLogit.lambda(lambdas)
	} else if(object@K>2){
		lambdaMatrix <- matrix(lambdas,byrow=FALSE,ncol=object@K)
		piMatrix <- softmax.lambdaMatrix(lambdaMatrix)
	}
	colnames(piMatrix) <- paste0('comp.',1:object@K)
	if(modal){
		getModalCluster(piMatrix)
	} else {
		piMatrix
	}
}
)

setGeneric('posterior',function(object,...){standardGeneric('posterior')})
setMethod('posterior','bDif',function(object,chains=object@chain.max,modal=FALSE){
	piMatrix <- clusters(object,chains=chains,modal=FALSE)
	if(!modal){
		sumPi <- apply(piMatrix,2,sum)
		return(sumPi/sum(sumPi))
	}
	if(modal){
		return(prop.table(table(getModalCluster(piMatrix))))
	}
})

#Factor scores
setMethod('factor.scores',signature='bDif',function(object,chains=object@chain.max){
	thetas <- summary_chains(object,pars='theta',chains=chains)[,1]
	thetas
})

#WAIC
setMethod('waic',signature='bDif',function(x){
	waic(extract_log_lik(x))
})

#LOO
setMethod('loo',signature='bDif',function(x){
	loo(extract_log_lik(x))
})

#Log-posterior
setGeneric('lp',def = function(object){standardGeneric('lp')})
setMethod('lp','bDif',function(object){get_posterior_mean(object,pars=c('lp__'))})

#Beta summary
summaryBetas <- function(bDif,chains=bDif@chain.max){
	betasSum <- summary_chains(bDif,chains=chains,pars=c('betas_logit','pi_sigma'))
	betasSum
}
#Dif table ala dichoDif, or Dif decision probability, Delta, and ppDelta>50%
summaryDif <- function(bDif,probs=c(.975,.95,.75),label=FALSE,chains=bDif@chain.max){
	#Transform to % labels
	probNames <- paste0(probs*100,'%')
	probNamesCol <- paste0('bDif.',probNames)
	
	#95% >50 implies we need to look at 5% lower credible interval
	probsInv <- 1 - probs
	sumOut <- summary_chains(bDif,pars='delta_logit',probs=probsInv,chains=chains)[,paste0(probsInv*100,'%')]
	difTrue <- sumOut >= 0
	delta <- invLogit(summary_chains(bDif,pars='delta_logit',chains=chains)[,1])
	pp50 <- apply(extract(bDif,pars='delta_logit',permuted=FALSE)[,chains,,drop=FALSE],MARGIN=3,function(x){mean(x>=0)})
	tableColNames <- c(probNamesCol,'Delta','pp>50%')
	if(label){
		difTable <- difTrue
		difTable[difTrue] <- 'DIF'
		difTable[!difTrue] <- 'NoDIF'
		difTable <- cbind(difTable,paste0(round(100*delta,2),'%'),paste0(round(100*pp50,2),'%'))
		colnames(difTable) <- tableColNames
		difTable
	} else {
		difTable <- cbind(difTrue,delta,pp50)
		colnames(difTable) <- tableColNames
		difTable
	}
}

# Compare bDif to output from difR package
compareMethods <- function(object,chains=object@chain.max,groups=clusters(object,modal=TRUE,chains),focal.name=1,method = c('TID','BD','Raju','MH','Logistic','Std','Lord'),...){
	if(object@model.type != '2PL'){
		stop('Function only applies to 2PL dichotomous IRT models')
	}
	dichoOut <- dichoDif(object@data,groups,focal.name,method = method)
	dichoOut$DIF <- cbind(dichoOut$DIF,summaryDif(object,label=TRUE,chains=chains,...))
	dichoOut
}

#Coefficients; alphas, diffs, deltas, pp>50%; ltm estimates optional
setMethod('coef',signature = c(object='bDif'),function(object,chains=object@chain.max,ltm=TRUE,cut=NULL){
	if(object@model.type == '2PL'){
		alphas <- summary_chains(object,pars='alpha',chains=chains)[,1]
		diffs <- summary_chains(object,pars='diff',chains=chains)[,1]
		difTable <- summaryDif(object)
		deltaMatrix <- difTable[,c('Delta','pp>50%')]
		## Non-dif model technique
		alpha_nondifs <- summary_chains(object,pars='alpha_nondif',chains=chains)[,1]
		diff_nondifs <- summary_chains(object,pars='diff_nondif',chains=chains)[,1]
		
		alphaMatrix <- matrix(alphas,byrow=TRUE,ncol=object@K)
		colnames(alphaMatrix) <- paste0('alpha.',1:object@K)
		
		diffMatrix <- matrix(diffs,byrow=TRUE,ncol=object@K)
		colnames(diffMatrix) <- paste0('diff.',1:object@K)
		
		#itemMatrix <- cbind(alphaMatrix,diffMatrix,deltaMatrix)
		if(!is.null(cut)){
			difItems <- deltaMatrix[,'pp>50%'] >= cut
			alphaMatrix[!difItems,] <- NA
			diffMatrix[!difItems,] <- NA
			alpha_nondifs[difItems] <- NA
			diff_nondifs[difItems] <- NA
		}
		itemMatrix <- cbind(alphaMatrix,'alpha..'=alpha_nondifs,diffMatrix,'diff..'=diff_nondifs,deltaMatrix)
		rownames(itemMatrix) <- 1:nrow(diffMatrix)
		
		if(ltm){
			ltmEst <- coef(ltm(object@data ~ z1,IRT.param = TRUE))[,c(2,1)]
			colnames(ltmEst) <- c('alpha.ltm','diff.ltm')
			if(!is.null(cut)){
				ltmEst[difItems,] <- NA
			}
			itemMatrix <- cbind(itemMatrix,ltmEst)
		}
		itemMatrix
	} else {stop('Only 2PL models currently supported.')}
}
)