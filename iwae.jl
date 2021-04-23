### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils
using Flux
using StatsFuns: log1pexp #log(1 + exp(x))
using Plots
using Statistics


### Load MNIST Digits and binarize
begin
	train_digits = Flux.Data.MNIST.images(:train)
	greyscale_MNIST = hcat(float.(reshape.(train_digits,:))...)
	binarized_MNIST = greyscale_MNIST .> 0.5
	BS = 200
	batches = Flux.Data.DataLoader(binarized_MNIST, batchsize=BS)
end

# log functions
function bernoulli_log_density(x, logit_means)
  """Numerically stable log_likelihood under bernoulli by accepting μ/(1-μ)"""
	b = x .* 2 .- 1 # [0,1] -> [-1,1]
  	return - log1pexp.(-b .* logit_means)
end

function factorized_gaussian_log_density(samples, μ, logσ)
	σ = exp.(logσ)
	return sum(-0.5*((samples.-μ)./σ).^2 .- log.(σ.*sqrt(2π)),dims=1)
end

sample_from_var_dist(μ, logσ) = (randn(size(μ)) .* exp.(logσ) .+ μ)

log_prior(z) = factorized_gaussian_log_density(z, 0, 0)

log_q(z, q_μ, q_logσ) = factorized_gaussian_log_density(z, q_μ, q_logσ)

function log_likelihood_iwae(x,z)
	""" Compute log likelihood log_p(x|z)"""
	return sum(bernoulli_log_density(x, decoder_iwae(z)),dims=1)
end

joint_log_density_iwae(x,z) = log_prior(z) .+ log_likelihood_iwae(x,z)






# Encoder and Decoder
Dz, Dh, Ddata = 2, 500, 28^2

function unpack_guassian_params(output)
	μ, logσ = output[1:2,:], output[3:4,:]
	return μ, logσ
end

encoder_iwae = Chain(Dense(Ddata, Dh, tanh), Dense(Dh, Dz*2), unpack_guassian_params)
decoder_iwae = Chain(Dense(Dz, Dh, tanh), Dense(Dh, Ddata))




# IWAE paper: https://arxiv.org/pdf/1509.00519.pdf
function iw_elbo(x, k)
	mean, sigma = encoder_iwae(x)
	z = sample_from_var_dist(mean, sigma)
	joint_ll = joint_log_density_iwae(x,z)
	log_q_z = log_q(z, mean, sigma)
	log_weight = joint_ll - log_q_z

	for i in 1:k-1
		z_new = sample_from_var_dist(mean, sigma)
		joint_ll_new = joint_log_density_iwae(x,z_new)
		log_q_z_new = log_q(z_new, mean, sigma)
		log_weight_new = joint_ll_new - log_q_z_new
		log_weight = vcat(log_weight, log_weight_new)
	end

	# Normalize importance weights
	weight_normalized = softmax(log_weight)
	loss = sum(weight_normalized .* log_weight)/size(x)[2]
	return loss
end

function iwae_loss(x, k)
	return -iw_elbo(x, k)
end

function train_iwae!(enc, dec, data; nepochs=100, k=5)
	params = Flux.params(enc, dec)
	opt = ADAM()
	@info "Begin training with k = $k"
	for epoch in 1:nepochs
		b_loss = 1000
		for batch in data
			grads = Flux.gradient(params) do
				loss = iwae_loss(batch, k)
				if b_loss > loss
					b_loss = loss
				end
				return loss
			end
			Flux.Optimise.update!(opt,params,grads) # update parameters
		end
		@info "Epoch $epoch: loss:$b_loss"
	end
	@info "Training is done"
end





# k = 5
train_iwae!(encoder_iwae, decoder_iwae, batches, nepochs=5)

# k = 10
train_iwae!(encoder_iwae, decoder_iwae, batches, nepochs=5, k=10)





# Visualization
bernoulli(p) = rand(Float64, size(p)) .< p

function iwae_mean_smaple_images(x)
	q_μ, q_logσ = encoder_iwae(x)
	z = randn(2,) .* exp.(q_logσ) .+ q_μ
	logit = decoder_iwae(z)
	bern = exp.(logit)./(1 .+ exp.(logit))
	sample = bernoulli(bern)

	# push the grayscale images to plots
	mean_p = heatmap(reshape(bern,(28,28)), color =:grays,
			         title = "mean", framestyle = :none,
					 aspect_ratio=:equal, legend=false)

	sample_p = heatmap(reshape(sample,(28,28)), color =:grays,
					   title = "sample", framestyle = :none,
					   aspect_ratio=:equal, legend=false)
	
	return mean_p, sample_p
end

begin
	# Random 4 digits in the first batch
	digits_index = [2, 9, 18]
	
	plots2 = []
	for index in digits_index
		x = first(batches)[:,index]
		x_p = heatmap(reshape(x,(28,28)), color =:grays,
		              title = "digit", framestyle = :none,
		              aspect_ratio=:equal, legend=false)
		push!(plots2, x_p)
		
		# IWAE: sample 1 z and 1 x
		iwae_mean, iwae_sample = iwae_mean_smaple_images(x)
		push!(plots2, iwae_mean)
		push!(plots2, iwae_sample)
	end

	# plot the plots in a 4 x 5 grid
	plot(plots2 ..., layout = (3, 3), size=(300, 400))
end

begin
	# get the train labels for the first batch
	train_labels = Flux.Data.MNIST.labels(:train)[1:BS]
	μ, logσ = encoder_iwae(first(batches))
	scatter(μ[1,:], μ[2,:], group=train_labels, xlabel="latent 1", ylabel="latent 2", title="mean vectors wrt digit labels")
end

savefig("IWAE_visualization_k=10.png")