data {
	int<lower=0> N;
	int<lower=0> J;
	int<lower=0> K;
	int L;
	int<lower=0,upper=J> jOrder;
	matrix[N,L] covariates;
	int<lower=0,upper=1> y[N,J];
	
}
transformed data{
	int D;
	D = 2;
}
parameters {
	//Item parameters
	vector<lower=0>[J] alpha[K];
	vector[J-1] diff_base[K];
	ordered[K] diff_ordered;
	vector[J] delta_logit_s;
	//Person parameters
	vector[N] theta;
	vector[K-1] pi_logit_s[N];
	//Softmax parameters
	matrix[L,K-1] beta_logit;
	
}
transformed parameters{
	vector[J] diff[K];
	vector[J] delta_logit;
	vector[D] delta[J];
	matrix[L,K] betas_logit;
	vector[K] pi_logit[N];
	vector[K] pi[N];
	
	delta_logit = delta_logit_s*3;
	//softmax betas construction
	betas_logit[1:L,1] = rep_vector(0,L);
	for(l in 1:L){
		betas_logit[l,2:K] = beta_logit[l];
	}
	//Diff matrix construction
	for(k in 1:K){
		int skip;
		skip = 0;
		diff[k,jOrder] = diff_ordered[k];
		for(j in 1:(J-1)){
			if(j == jOrder){
				skip = skip+1;
				}
			diff[k,j+skip] = diff_base[k,j];
		}
	}
	//Delta_logit conversion
	for(j in 1:J){
		delta[j,1] = log_inv_logit(-delta_logit[j]);
		delta[j,D] = log_inv_logit(delta_logit[j]);
	}
	//Pi_logit conversion
	for(n in 1:N){
		pi_logit[n,1] = 0;
		pi_logit[n,2:K] = pi_logit_s[n]*3 + (covariates[n]*betas_logit[1:L,2:K])';
		pi[n] = log_softmax(pi_logit[n]);
	}
	
}
model {
	//lp declarations
	vector[K] lp_jndk[J,N];
	vector[D] lp_jnd[J,N];
	vector[N] lp_jn[J];
	vector[J] lp_j;
	
	//Priors
	////Item parameters
	for(k in 1:K){
		diff_base[k] ~ normal(0,1);
		alpha[k] ~ normal(0,1);
	}
	diff_ordered ~ normal(0,1);
	delta_logit_s ~ logistic(0,1);
	
	////Softmax parameters
	for(k in 1:(K-1)){
		beta_logit'[k] ~ logistic(0,1);
	}
	////Person parameters
	theta ~ normal(0,1);
	for(n in 1:N){
		pi_logit_s[n,1:(K-1)] ~ logistic(0,1);
	}
	
	//Likelihood
	for(j in 1:J){
		for(n in 1:N){
			//lp_jnd[j,n,1] = delta[j,1] + bernoulli_logit_lpmf(y[n,j] | alpha_nondif[j]*(theta[n] - diff_nondif[j]));
			lp_jnd[j,n,1] = delta[j,1] + bernoulli_logit_lpmf(y[n,j] | alpha[1,j]*(theta[n] - diff[1,j]));
			for(k in 1:K){
				lp_jndk[j,n,k] = pi[n,k] + bernoulli_logit_lpmf(y[n,j] | alpha[k,j]*(theta[n] - diff[k,j]));
			}
			lp_jnd[j,n,2] = delta[j,2] + log_sum_exp(lp_jndk[j,n]);
			lp_jn[j,n] = log_sum_exp(lp_jnd[j,n]);
		}
		lp_j[j] = sum(lp_jn[j]);
	}
	target += sum(lp_j);
}
generated quantities{
	real log_lik[N];
	vector[K] lp_jndk[J,N];
	vector[D] lp_jnd[J,N];
	vector[N] lp_jn[J];
	for(j in 1:J){
		for(n in 1:N){
			lp_jnd[j,n,1] = delta[j,1] + bernoulli_logit_lpmf(y[n,j] | alpha[1,j]*(theta[n] - diff[1,j]));
			for(k in 1:K){
				lp_jndk[j,n,k] = pi[n,k] + bernoulli_logit_lpmf(y[n,j] | alpha[k,j]*(theta[n] - diff[k,j]));
			}
			lp_jnd[j,n,2] = delta[j,2] + log_sum_exp(lp_jndk[j,n]);
			lp_jn[j,n] = log_sum_exp(lp_jnd[j,n]);
		}
	}
	for(n in 1:N){
		log_lik[n] = sum(lp_jn[1:J,n]);
	}
}
