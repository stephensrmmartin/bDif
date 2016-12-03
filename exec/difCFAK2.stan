data {
	int N;
	int J;
	int L;
	int jOrder;
	matrix[N,L] covariates;
	matrix[N,J] y;
	
}

transformed data{
	int K;
	int D;
	K = 2;
	D = 2;
	
}

parameters {
	//Item parameters
	## change to 0 for non-DIF, since data are scaled?
	vector[J] intercept[K];
	//vector[J-1] intercept_base[K];
	//ordered[K] intercept_ordered;
	vector<lower=0>[J-1] lambda_base[K];
	vector<lower=0>[K] lambda_ordered;
	vector<lower=0>[J] residual_s[K];
	vector[J] delta_logit_s;
	
	//Person parameters
	vector[N] theta;
	vector[N] pi_logit_s;
	
	//Softmax parameters
	vector[L] betas_logit;
	
}

transformed parameters {
	//vector[J] intercept[K];
	vector[J] lambda[K];
	vector[J] delta_logit;
	vector[N] pi_logit;
	vector<lower=0>[J] residual[K];
	
	pi_logit = pi_logit_s + covariates * betas_logit;
	
	delta_logit = delta_logit_s * 3;
	
	//Intercept matrix construction
	for(k in 1:K){
//		intercept[k,jOrder] = intercept_ordered[k];
		lambda[k,jOrder] = lambda_ordered[k];
	}
	{
		int skip;
		skip = 0;
		
		for(j in 1:(J-1)){
			if(j == jOrder){skip = skip + 1;}
			lambda[1,j + skip] = lambda_base[1,j];
			lambda[2,j + skip] = lambda_base[2,j];
		}
	}
	
	for(k in 1:K){
		residual[k] = residual_s[k] .* residual_s[k];
	}
	
}

model {
	//Declarations
	vector[K] yhat[J,N];
	vector[2] lp_jnd[J,N];
	vector[2] lp_jndk[J,N];
	vector[N] lp_jn[J];
	vector[J] lp_j;
	
	//Item priors
	for(k in 1:K){
		//intercept_base[k] ~ normal(0,1);
		intercept[k] ~ normal(0,1);
		lambda_base[k] ~ lognormal(0,1);
		residual_s[k] ~ normal(0,1);
	}
	lambda_ordered ~ lognormal(0,1);
	delta_logit_s ~ logistic(0,1);
	//Person priors
	theta ~ normal(0,1);
	pi_logit_s ~ logistic(0,1);
	//Softmax priors
	betas_logit ~ logistic(0,1);
	
	//Likelihood
	for(j in 1:J){
		for(n in 1:N){
			for(k in 1:K){
				yhat[j,n,k] = intercept[k,j] + lambda[k,j]*theta[n];
			}
			lp_jnd[j,n,1] = log_inv_logit(-delta_logit[j]) + normal_lpdf(y[n,j] | yhat[j,n,1], residual_s[1,j]);
			lp_jndk[j,n,1] = log_inv_logit(-pi_logit[n]) + normal_lpdf(y[n,j] | yhat[j,n,1], residual_s[1,j]);
			lp_jndk[j,n,2] = log_inv_logit(pi_logit[n]) + normal_lpdf(y[n,j] | yhat[j,n,2], residual_s[2,j]);
			lp_jnd[j,n,2] = log_inv_logit(delta_logit[j]) + log_sum_exp(lp_jndk[j,n]);
			lp_jn[j,n] = log_sum_exp(lp_jnd[j,n]);
		}
		lp_j[j] = sum(lp_jn[j]);
	}
	target += sum(lp_j);
	
}

generated quantities {
	vector[K] yhat[J,N];
	vector[2] lp_jnd[J,N];
	vector[2] lp_jndk[J,N];
	vector[N] lp_jn[J];
	vector[N] log_lik;
	for(j in 1:J){
		for(n in 1:N){
			for(k in 1:K){
				yhat[j,n,k] = intercept[k,j] + lambda[k,j]*theta[n];
			}
			lp_jnd[j,n,1] = log_inv_logit(-delta_logit[j]) + normal_lpdf(y[n,j] | yhat[j,n,1], residual[1,j]);
			lp_jndk[j,n,1] = log_inv_logit(-pi_logit[n]) + normal_lpdf(y[n,j] | yhat[j,n,1], residual[1,j]);
			lp_jndk[j,n,2] = log_inv_logit(pi_logit[n]) + normal_lpdf(y[n,j] | yhat[j,n,2], residual[2,j]);
			lp_jnd[j,n,2] = log_inv_logit(delta_logit[j]) + log_sum_exp(lp_jndk[j,n]);
			lp_jn[j,n] = log_sum_exp(lp_jnd[j,n]);
		}
	}
	for(n in 1:N){
		log_lik[n] = sum(lp_jn[1:J,n]);
	}
	
}
