data{
	int N;
	int J;
	int L;
	int jOrder;
	matrix[N,L] covariates;
	int y[N,J];
}
transformed data{
	int K;
	int D;
//	real<lower=0> pi_sigma;
	K = 2;
	D = 2;
//	pi_sigma = 1;
}
parameters{
	//Item parameters
	vector<lower=0>[J] alpha[K];
	vector[J-1] diff_base[K];
	ordered[K] diff_ordered;
	vector[J] delta_logit_s;
	//Person parameters
	vector[N] theta;
	vector[N] pi_logit_s;
	//Softmax/logistic parameters
	vector[L] beta_logit;
	
}
transformed parameters{
	vector[L] betas_logit;
	vector[J] diff[K];
	vector[J] delta_logit;
	vector[N] pi_logit;
	delta_logit = delta_logit_s * 3;
	betas_logit = beta_logit;
	pi_logit = pi_logit_s*3 + covariates*betas_logit;
	for(k in 1:K){
		diff[k,jOrder] = diff_ordered[k];
	}
	{
		int skip;
		skip = 0;
		for(j in 1:(J-1)){
			if(j == jOrder) skip = skip + 1;
			diff[1,j+skip] = diff_base[1,j];
			diff[2,j+skip] = diff_base[2,j];
		}
	}
	
}
model{
	vector[J] lp_j;
	vector[2] lp_jnd[J,N];
	vector[2] lp_jndk[J,N];
	vector[N] lp_jn[J];
	//Item priors
	for(k in 1:K){
		alpha[k] ~ normal(0,1);
		diff_base[k] ~ normal(0,1);
	}
	diff_ordered ~ normal(0,1);
	delta_logit_s ~ logistic(0,1);
	
	//Softmax priors
	beta_logit ~ logistic(0,1);
	//Person priors
	theta ~ normal(0,1);
	pi_logit_s ~ logistic(0, 1);
	
	
	//likelihood -- \delta_j per person
	for(j in 1:J){
		for(n in 1:N){
			lp_jnd[j,n,1] = log_inv_logit(-delta_logit[j]) + bernoulli_logit_lpmf(y[n,j] | alpha[1,j]*(theta[n] - diff[1,j]));
			//lp_jnd[j,n,1] = log_inv_logit(-delta_logit[j]) + bernoulli_logit_lpmf(y[n,j] | alpha_nondif[j]*(theta[n] - diff_nondif[j]));
			lp_jndk[j,n,1] = log_inv_logit(-pi_logit[n]) + bernoulli_logit_lpmf(y[n,j] | alpha[1,j]*(theta[n] - diff[1,j]));
			lp_jndk[j,n,2] = log_inv_logit(pi_logit[n]) + bernoulli_logit_lpmf(y[n,j] | alpha[2,j]*(theta[n] - diff[2,j]));
			lp_jnd[j,n,2] = log_inv_logit(delta_logit[j]) + log_sum_exp(lp_jndk[j,n]);
			lp_jn[j,n] = log_sum_exp(lp_jnd[j,n]);
		}
		lp_j[j] = sum(lp_jn[j]);
	}
	target += sum(lp_j);
}
generated quantities{
	real log_lik[N];
	vector[2] lp_jnd[J,N];
	vector[2] lp_jndk[J,N];
	vector[N] lp_jn[J];
	for(j in 1:J){
		for(n in 1:N){
			lp_jnd[j,n,1] = log_inv_logit(-delta_logit[j]) + bernoulli_logit_lpmf(y[n,j] | alpha[1,j]*(theta[n] - diff[1,j]));
			//jndk is actually J,N,K=2; 
			lp_jndk[j,n,1] = log_inv_logit(-pi_logit[n]) + bernoulli_logit_lpmf(y[n,j] | alpha[1,j]*(theta[n] - diff[1,j]));
			lp_jndk[j,n,2] = log_inv_logit(pi_logit[n]) + bernoulli_logit_lpmf(y[n,j] | alpha[2,j]*(theta[n] - diff[2,j]));
			lp_jnd[j,n,2] = log_inv_logit(delta_logit[j]) + log_sum_exp(lp_jndk[j,n]);
			lp_jn[j,n] = log_sum_exp(lp_jnd[j,n]);
		}
	}
	for(n in 1:N){
		log_lik[n] = sum(lp_jn[1:J,n]);
	}
}
