#' @import Rcpp
#' @useDynLib bDif, .registration=TRUE
NULL

# bDif class functions ------------------ 
# Class creation
#' An S4 class representing the Bayesian DIF/measurement invariance model
#' 
#' Inherits from stanfit
#' @slot data The response matrix implied by the measurement model.
#' @slot K The number of latent groups estimated.
#' @slot model.type The measurement model type (2PL and CFA currently supported)
#' @slot chain.max The mcmc chain with the highest log posterior probability.
#' @export
bDif <- setClass('bDif',contains = 'stanfit',slots = c('data','K','model.type','chain.max'))

#' Fit the Bayesian DIF/measurement invariance model
#' 
#' Fit the Bayesian DIF/measurement invariance model.
#' The model assumes that K latent groups exist between which some set of
#' items operate differentially.
#' The model also, currently, assumes a dichotomous state of DIF, 
#' such that DIF either exists for some item j or not.
#' Finally, one can predict latent group membership from known groups or other covariates.
#' The model jointly estimates item parameters, the probability of DIF for each item, 
#' the probability of each individual belonging to each latent group, and latent abilities.
#' 
#' @param data A data.frame containing the indicators and any covariates.
#' @param measurementModel A RHS formula specifying the indicators for the latent factor.
#' @param K The number of latent groups to estimate.
#' @param order The item number whose difficulty (intercept) is ordered across latent groups.
#' @param covariateModel A RHS formula specifying the concomittant predictors of latent groups.
#' @param model.type A character string indicating the type of model to fit, either '2PL' or 'CFA'
#' @param method A character string indicating whether to use MCMC ('mcmc') or variational Bayes ('vb'). 'mcmc' is recommended.
#' @param ... Arguments passed to \code{\link[rstan]{sampling}} or \code{\link[rstan]{vb}}
#' @export
#' @return bDif S4 object. See \link{bDif}
bDifFit <- function(data, measurementModel, K, order,covariateModel = ~ 1, model.type = '2PL',method = 'mcmc',...){
	responseMatrix <- model.matrix(measurementModel,data)[,-1]
	responseMissing <- !complete.cases(responseMatrix)
	
	covariateMatrix <- model.matrix(covariateModel,data)
	covariateMissing <- !complete.cases(covariateMatrix)
	
	responseMatrix <- responseMatrix[!responseMissing & !covariateMissing,,drop=FALSE]
	if(sum(responseMissing)>0){
		warning('Removing ',sum(responseMissing),' cases missing response data.')
	}
	
	covariateMatrix <- covariateMatrix[!covariateMissing & !responseMissing,,drop=FALSE]
	if(sum(covariateMissing)>0){
		warning('Removing ',sum(covariateMissing),' cases missing covariate data.')
	}
	
	N <- nrow(responseMatrix)
	L <- ncol(covariateMatrix)
	J <- ncol(responseMatrix)
	
	stan_data <- list(y=responseMatrix,covariates=covariateMatrix,K=K,jOrder=order,L=L,N=N,J=J)
	
	if(model.type == '2PL'){
		pars <- c('alpha','diff','delta_logit','pi_logit','theta','betas_logit','log_lik')
		if(K == 2){
			model <- stanmodels$dif2PLK2
		} else {
			model <- stanmodels$dif2PLK3
		}
	} else if (model.type == 'CFA'){
		pars <- c('lambda','intercept','residual','delta_logit','pi_logit','theta','betas_logit')
		if(K == 2){
			model <- stanmodels$difCFAK2
		} else {
			stop('K=2 only supported currently for CFA')
			#model <- stanmodels$difCFAK3
		}
	} else {
		stop('model.type must be one of 2PL or CFA')
	}
	
	if(method=='mcmc'){
		stanOut <- rstan::sampling(object = model,data = stan_data,pars=pars,...)
	} else if(method=='vb'){
		stanOut <- rstan::vb(object = model,data = stan_data,pars=pars,algorithm='fullrank',...)
	}
		
	bDifOut <- bDif(stanOut,K=K,model.type=model.type,data=responseMatrix,chain.max=which.max(get_posterior_mean(stanOut,pars='lp__')))
	
	bDifOut
}

#Cluster membership
#' Generic function for extracting cluster memberships or probabilities.
#' 
#' Generic function for extracting cluster memberships or probabilities.
#' 
#' @param object Object containing cluster membership.
#' @param ... Arguments passed to object methods.
#' @export
setGeneric('clusters',function(object,...){standardGeneric('clusters')})

#' Obtain cluster membership assignments or probabilities for each individual.
#' 
#' By default, \code{clusters} will return the group each individual most probably belongs to.
#' If you want membership probabilities, change the \code{modal} argument.
#' 
#' @param object A bDif object returned from \code{\link{bDifOut}}.
#' @param chains A numeric vector indicating the mcmc chain(s).
#' @param modal Logical. Whether to return the modal group membership vector or a matrix of cluster membership probabilities.
#' @export
#' @return Either a vector of length N containing modal cluster memberships (modal=TRUE) or an NxK matrix
#'   containing cluster membership probabilities.
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

#' Generic Method. Obtain posterior probabilities of cluster membership
#' 
#' @param object Object containing cluster membership probabilities
#' @param ... Further arguments for non-generic methods.
#' @export
setGeneric('posterior',function(object,...){standardGeneric('posterior')})

#' Compute posterior probabilities of cluster membership.
#' 
#' Computes posterior probabilities in one of two ways.
#' One (modal=FALSE) calculates the marginalized probability of cluster membership
#' across all cases.
#' The other (modal=TRUE) calculates the proportions of individuals in each cluster.
#' 
#' @param object bDif object from bDifFit.
#' @param chains Numeric vector. Specifies which mcmc chains to use for calculation.
#' @param modal Whether to compute posterior probabilities based on modal assignments (TRUE)
#'   or to compute based on marginal probabilities.
#' @return A vector of length K containing the posterior probabilities of each cluster or component.
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
#' Compute factor scores for a latent variable model.
#' 
#' @param object Object from a latent variable model model fit.
#' @export
setGeneric('factor.scores',function(object,...){standardGeneric('factor.scores')})

#' Compute factor scores for cases in bDif object.
#' 
#' @param object bDif object from bDifFit.
#' @param chains Numeric vector. Which mcmc chain(s) to extract from.
#' @export
#' @return A vector of length N containing the posterior mean estimates of latent ability.
setMethod('factor.scores',signature='bDif',function(object,chains=object@chain.max){
	thetas <- summary_chains(object,pars='theta',chains=chains)[,1]
	thetas
})

#' Widely applicable information criterion (WAIC)
#' 
#' Widely applicable information criterion (WAIC)
#' 
#' @param x bDif object from bDifFit.
#' @importFrom loo waic
#' @export
waic.bDif <- function(x){
	loo::waic(loo::extract_log_lik(x))
}

#LOO
#' Leave-one-out cross-validation (LOO)
#' 
#' Efficient approximate leave-one-out cross-validation for Bayesian models.
#' 
#' @param x bDif object from bDifFit.
#' @importFrom loo loo
#' @export
loo.bDif <- function(x){
	loo::loo(loo::extract_log_lik(x))
}

#Log-posterior
setGeneric('lp',def = function(object){standardGeneric('lp')})
setMethod('lp','bDif',function(object){rstan::get_posterior_mean(object,pars=c('lp__'))})

#Beta summary
summaryBetas <- function(bDif,chains=bDif@chain.max){
	betasSum <- summary_chains(bDif,chains=chains,pars='betas_logit')
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
#' Compare DIF detection with methods from difR package
#' 
#' @inheritParams clusters,bDif-method
#' @param groups Vector containing group memberships. See \link[difR]{dichoDif}.
#' @param focal.name Name of reference group. See \link[difR]{dichoDif}.
#' @param method Character vector. Vector of DIF detection methods from the difR package. See \link[difR]{dichoDif}.
#' @param ... Not currently used.
#' @export
#' @return A dichoDif object. See \link[difR]{dichoDif}
compareMethods <- function(object,chains=object@chain.max,groups=clusters(object,modal=TRUE,chains),focal.name=1,method = c('TID','BD','Raju','MH','Logistic','Std','Lord'),...){
	if(object@model.type != '2PL'){
		stop('Function only applies to 2PL dichotomous IRT models')
	}
	dichoOut <- difR::dichoDif(object@data,groups,focal.name,method = method)
	dichoOut$DIF <- cbind(dichoOut$DIF,summaryDif(object,label=TRUE,chains=chains,...))
	dichoOut
}

#Coefficients; alphas, diffs, deltas, pp>50%; ltm estimates optional
#' Obtain item parameter estimates
#' 
#' Obtain item parameter estimates
#' @inheritParams clusters,bDif-method
#' @param include Logical. Whether to include estimates from ltm (model.type=2PL) or lavaan (model.type=CFA)
#' @param cut Numeric between 0 and 1. If pp>50% is greater than this value, the non-DIF item parameters
#'   are cut and the DIF item parameters retained. If lesser than this value, the opposite.
#' @export
#' @import ltm
#' @import lavaan
setMethod('coef',signature = c(object='bDif'),function(object,chains=object@chain.max,include=TRUE,cut=NULL){
	if(object@model.type == '2PL'){
		coef.2pl(object,chains,include,cut)
	} else if(object@model.type == 'CFA'){
		coef.cfa(object,chains,include,cut)
	}
}
)

coef.2pl <- function(object,chains=object@chain.max,include=TRUE,cut=NULL){
		alphas <- summary_chains(object,pars='alpha',chains=chains)[,1]
		diffs <- summary_chains(object,pars='diff',chains=chains)[,1]
		difTable <- summaryDif(object,chains=chains)
		deltaMatrix <- difTable[,c('Delta','pp>50%')]
		## Non-dif model technique
		#alpha_nondifs <- summary_chains(object,pars='alpha_nondif',chains=chains)[,1]
		#diff_nondifs <- summary_chains(object,pars='diff_nondif',chains=chains)[,1]
		
		alphaMatrix <- matrix(alphas,byrow=TRUE,ncol=object@K)
		colnames(alphaMatrix) <- paste0('alpha.',1:object@K)
		
		diffMatrix <- matrix(diffs,byrow=TRUE,ncol=object@K)
		colnames(diffMatrix) <- paste0('diff.',1:object@K)
		
		if(!is.null(cut)){
			difItems <- deltaMatrix[,'pp>50%'] >= cut
			## If using non-dif formulation, uncomment.
			#alphaMatrix[!difItems,] <- NA
			#diffMatrix[!difItems,] <- NA
			#alpha_nondifs[difItems] <- NA
			#diff_nondifs[difItems] <- NA
			alphaMatrix[!difItems,2:object@K] <- NA
			diffMatrix[!difItems,2:object@K] <- NA
		}
		##If using non-dif formulation, uncomment
		#itemMatrix <- cbind(alphaMatrix,'alpha..'=alpha_nondifs,diffMatrix,'diff..'=diff_nondifs,deltaMatrix)
		itemMatrix <- cbind(alphaMatrix,diffMatrix,deltaMatrix)
		rownames(itemMatrix) <- colnames(object@data)
		
		if(include){
			ltmEst <- coef(ltm::ltm(object@data ~ z1,IRT.param = TRUE))[,c(2,1)]
			colnames(ltmEst) <- c('alpha.ltm','diff.ltm')
			if(!is.null(cut)){
				ltmEst[difItems,] <- NA
			}
			itemMatrix <- cbind(itemMatrix,ltmEst)
		}
		itemMatrix
	
}

coef.cfa <- function(object,chains=object@chain.max,include=TRUE,cut=NULL){
	intercept = summary_chains(object,chains=chains,pars='intercept')[,1]
	lambda = summary_chains(object,chains=chains,pars='lambda')[,1]
	residual = summary_chains(object,chains=chains,pars='residual')[,1]
	
	#Intercept matrix
	interceptMatrix <- matrix(intercept,byrow=TRUE,ncol=object@K)
	colnames(interceptMatrix) <- paste0('int.',1:object@K)
	
	#Loading matrix
	lambdaMatrix <- matrix(lambda,byrow=TRUE,ncol=object@K)
	colnames(lambdaMatrix) <- paste0('lam.',1:object@K)
	
	#Residual matrix
	residualMatrix <- matrix(residual,byrow=TRUE,ncol=object@K)
	colnames(residualMatrix) <- paste0('res.',1:object@K)
	
	#Delta matrix
	deltaMatrix <- summaryDif(object,chains=chains)[,c('Delta','pp>50%')]
	
	itemMatrix <- cbind(interceptMatrix,lambdaMatrix,residualMatrix,deltaMatrix)
	rownames(itemMatrix) <- colnames(object@data)
	
	## lavaan estimates
	if(include){
		d <- as.data.frame(object@data)
		varnames <- names(d)
		lavRHS <- paste(varnames,collapse = ' + ')
		lavModel <- paste0('f1 =~ ', lavRHS)
		lavOut <- cfa(data=d, model = lavModel, std.lv = TRUE)
		lavTab <- parameterEstimates(lavOut)
		lavIntercept <- lavTab[lavTab[,2] == '~1','est']
		lavLoading <- lavTab[lavTab[,2] == '=~','est']
		lavResidual <- lavTab[lavTab[,2] == '~~' & lavTab[,1] == lavTab[,3], 'est']
		lavMatrix <- cbind('int.lav' = lavIntercept, 'lam.lav' = lavLoading,'res.lav' = lavResidual)
		itemMatrix <- cbind(itemMatrix, lavMatrix)
	}
	
	itemMatrix
	
}