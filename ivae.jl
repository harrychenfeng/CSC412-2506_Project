### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ e5416a10-a01c-11eb-1cab-793d2df3d6c1
using Flux

# ╔═╡ 2f9e8ec8-dcf7-4e8c-b7ac-2a15f90fc60c
using StatsFuns: log1pexp #log(1 + exp(x))

# ╔═╡ 7d758b6c-9cfd-4085-bd84-230e91f8190b
using BSON: @load

# ╔═╡ 1eb02d21-8125-4955-9cd6-6c0a0ac4b586
using BSON: @save

# ╔═╡ 5f3207d1-b95d-4038-b708-1ceb421cca5d
using Plots

# ╔═╡ 3fad4b3c-d4ea-443c-b64d-18b9e127d3cf
using Statistics

# ╔═╡ ce104a6c-a383-4d04-a81e-45204f873ac2
md"""
### Load MNIST Digits
"""

# ╔═╡ f9f90fdf-c8a3-4478-b9ca-76904c3b1d91
begin
	train_digits = Flux.Data.MNIST.images(:train)
	greyscale_MNIST = hcat(float.(reshape.(train_digits,:))...)
	binarized_MNIST = greyscale_MNIST .> 0.5
	BS = 200
	batches = Flux.Data.DataLoader(binarized_MNIST, batchsize=BS)
end

# ╔═╡ e4951f61-99ee-4295-ae51-c0deffd3c66a
md"""
### Log Density, Encoder and Decoder
"""

# ╔═╡ c11be4e5-644d-41a2-9672-4582bfb771d4
function bernoulli_log_density(x, logit_means)
  """Numerically stable log_likelihood under bernoulli by accepting μ/(1-μ)"""
	b = x .* 2 .- 1 # [0,1] -> [-1,1]
  	return - log1pexp.(-b .* logit_means)
end

# ╔═╡ 59e7823a-1a15-4b4a-8675-7bb2ffaa3dc6
function factorized_gaussian_log_density(samples, μ, logσ)
	σ = exp.(logσ)
	return sum(-0.5*((samples.-μ)./σ).^2 .- log.(σ.*sqrt(2π)),dims=1)
end

# ╔═╡ df7731fe-81c3-4b7d-b782-35ff8746adf9
log_prior(z) = factorized_gaussian_log_density(z, 0, 0)

# ╔═╡ cf3ebd63-c0f8-4478-a132-6b7a95897ab2
Dz, Dh, Ddata = 2, 500, 28^2

# ╔═╡ f87dd3c5-d4cd-4d0f-b317-3f56cbaaa61d
log_q(z, q_μ, q_logσ) = factorized_gaussian_log_density(z, q_μ, q_logσ)

# ╔═╡ 5611074c-082e-42fa-91de-3ac6553b0160
function unpack_guassian_params(output)
	μ, logσ = output[1:2,:], output[3:4,:]
	return μ, logσ
end

# ╔═╡ 2bde411e-db3b-42e5-aa6a-909c430d0c6c
sample_from_var_dist(μ, logσ) = (randn(size(μ)) .* exp.(logσ) .+ μ)

# ╔═╡ deaad4a8-c5af-4a71-833f-735d2ef6d11d
function log_likelihood(x,z)
  """ Compute log likelihood log_p(x|z)"""
	# use numerically stable bernoulli
	return sum(bernoulli_log_density(x, decoder(z)),dims=1)
end

# ╔═╡ 9a5fa88c-65c6-472a-9955-75be909319e0
joint_log_density(x,z) = log_prior(z) .+ log_likelihood(x,z)

# ╔═╡ e1b6014d-0425-48d1-9e3e-841fe7fbfa7f
encoder = Chain(Dense(Ddata, Dh, tanh), Dense(Dh, Dz*2), unpack_guassian_params)

# ╔═╡ e5137cdc-246b-474d-8b52-cb1bf7eb1c24
decoder = Chain(Dense(Dz, Dh, tanh), Dense(Dh, Ddata))

# ╔═╡ a1ead5c7-8a2e-4eda-ace2-fe44c9f55c19
function elbo(x)
  #TODO variational parameters from data	
  q_μ, q_logσ = encoder(x)
  #TODO: sample from variational distribution
  z = sample_from_var_dist(q_μ, q_logσ)
  #TODO: joint likelihood of z and x under model
  joint_ll = joint_log_density(x,z)
  #TODO: likelihood of z under variational distribution
  log_q_z = log_q(z, q_μ, q_logσ)
  #TODO: Scalar value, mean variational evidence lower bound over batch
  elbo_estimate = sum(joint_ll - log_q_z)/size(x)[2]
  return elbo_estimate
end

# ╔═╡ 78a21a4f-925c-4029-a7e2-441f74aaf407
function loss(x)
  return -elbo(x)
end

# ╔═╡ b353df2e-1197-45da-b44e-22133be729e3
md"""
### Train 2D model
"""

# ╔═╡ 98af9656-8fd7-450c-af9c-f97ec13156dd
function train!(enc, dec, data; nepochs=100)
	params = Flux.params(enc, dec)
	opt = ADAM()
	@info "Begin training in 2D latent space"
	for epoch in 1:nepochs
		b_loss = 0
		for batch in data
			# compute gradient wrt loss
			grads = Flux.gradient(params) do
				b_loss = loss(batch)
				return b_loss
			end
			# update parameters
			Flux.Optimise.update!(opt, params, grads)
		end
		# Optional: log loss using @info "Epoch $epoch: loss:..."
		@info "Epoch $epoch: loss:$b_loss"
		# Optional: visualize training progress with plot of loss
	end
	@info "Training in 2D is done"
	# return nothing, this mutates the parameters of enc and dec!
end

# ╔═╡ 0fb26a86-d1aa-4b15-a3c2-681ba08343b9
# train!(encoder, decoder, batches, nepochs=3)

# ╔═╡ d8b579ee-575c-49bf-aafb-544347719a9c
md"""
### Train Importance Weighted AutoEncoder

This is the implementation of the importance weighted autoencoders (IWAE) proposed in [this paper](https://arxiv.org/pdf/1509.00519.pdf). I also looked at the [github code](https://github.com/xqding/Importance_Weighted_Autoencoders/blob/master/model/vae_models.py) and [github code](https://github.com/yburda/iwae/blob/master/iwae.py) for IWAE written in Python.
"""

# ╔═╡ 88118316-1b8f-4cd9-a8b7-0f8d925205dc
encoder_iwae = Chain(Dense(Ddata, Dh, tanh), Dense(Dh, Dz*2), unpack_guassian_params)

# ╔═╡ c07f07fc-642a-47e8-9ec8-4abda3d3a34d
decoder_iwae = Chain(Dense(Dz, Dh, tanh), Dense(Dh, Ddata))

function log_likelihood_iwae(x,z)
	""" Compute log likelihood log_p(x|z)"""
	return sum(bernoulli_log_density(x, decoder_iwae(z)),dims=1)
end

joint_log_density_iwae(x,z) = log_prior(z) .+ log_likelihood_iwae(x,z)

# ╔═╡ 647b08cf-0853-4c90-b33e-2b04548c5b2b
function iw_elbo(x)
	q_μ, q_logσ = encoder_iwae(x)
	z = sample_from_var_dist(q_μ, q_logσ)

	joint_ll = joint_log_density_iwae(x,z)
	log_q_z = log_q(z, q_μ, q_logσ)
	log_weight = joint_ll - log_q_z
	
	# Below are the extra codes for weight
	weight_minus_max = log_weight .- maximum(log_weight)
	weight_normalized = softmax(weight_minus_max)
	loss = sum(weight_normalized .* log_weight)/size(x)[2]
	return -loss
end

# ╔═╡ 64292978-40b6-4cef-ab62-768bfbf4854e
function iwae_loss(x)
	return -iw_elbo(x)
end

# ╔═╡ a4ca2198-5bab-4fae-ae14-ccb4d17c3186
function train_iwae!(enc, dec, data; nepochs=100, k=5)
	params = Flux.params(enc, dec)
	opt = ADAM()
	@info "Begin training in 2D latent space"
	for epoch in 1:nepochs
		b_loss = 0
		for batch in data
			# compute gradient wrt loss
			grads = Flux.gradient(params) do
				b_loss = iwae_loss(batch, k)
				return b_loss
			end
			# update parameters
			Flux.Optimise.update!(opt, params, grads)
		end
		# Optional: log loss using @info "Epoch $epoch: loss:..."
		@info "Epoch $epoch: loss:$b_loss"
		# Optional: visualize training progress with plot of loss
	end
	@info "Training in 2D is done"
	# return nothing, this mutates the parameters of enc and dec!
end

# ╔═╡ 7b949260-94e3-47ac-96b9-66fc73dfa234
begin
	train_iwae!(encoder_iwae, decoder_iwae, batches, nepochs=3)
	# train_iwae!(encoder_iwae, decoder_iwae, batches, nepochs=3, k=10)
	# train_iwae!(encoder_iwae, decoder_iwae, batches, nepochs=3, k=50)
end

# ╔═╡ 47d5146b-bd02-4347-8701-9ed68e3d3405
md"""
### Visualize IWAE with baseline VAE
"""

# ╔═╡ 7c675eeb-0b74-4346-9776-fbfbf72b7b8e
function vae_mean_smaple_images(x)
	q_μ, q_logσ = encoder(x)
	z = randn(2,) .* exp.(q_logσ) .+ q_μ
	logit = decoder(z)
	bern = exp.(logit)./(1 .+ exp.(logit))
	sample = bernoulli(bern)

	# push the grayscale images to plots
	mean_p = heatmap(reshape(bern,(28,28)), color =:grays,
			         title = "vae", framestyle = :none,
					 aspect_ratio=:equal, legend=false)

	sample_p = heatmap(reshape(sample,(28,28)), color =:grays,
			           framestyle = :none,
					   aspect_ratio=:equal, legend=false)
	
	return mean_p, sample_p
end

# ╔═╡ 958a7118-7283-4509-b1cd-30d2458e2489
bernoulli(p) = rand(Float64, size(p)) .< p

# ╔═╡ 137d48da-a8a2-43e2-94ea-36771aa4d9fa
function iwae_mean_smaple_images(x)
	q_μ, q_logσ = encoder_iwae(x)
	z = randn(2,) .* exp.(q_logσ) .+ q_μ
	logit = decoder_iwae(z)
	bern = exp.(logit)./(1 .+ exp.(logit))
	sample = bernoulli(bern)

	# push the grayscale images to plots
	mean_p = heatmap(reshape(bern,(28,28)), color =:grays,
			         title = "iwae", framestyle = :none,
					 aspect_ratio=:equal, legend=false)

	sample_p = heatmap(reshape(sample,(28,28)), color =:grays,
			           framestyle = :none,
					   aspect_ratio=:equal, legend=false)
	
	return mean_p, sample_p
end

# ╔═╡ d7f41eb6-48d4-47f4-8588-25f3accaf381
begin
	# Random 4 digits in the first batch
	digits_index = [1,2,7,8]
	
	plots2 = []
	for index in digits_index
		x = first(batches)[:,index]
		x_p = heatmap(reshape(x,(28,28)), color =:grays,
		              title = "digit", framestyle = :none,
		              aspect_ratio=:equal, legend=false)
		push!(plots2, x_p)

		# # VAE: sample 1 z and 1 x
		# vae_mean, vae_sample = vae_mean_smaple_images(x)
		# push!(plots2, vae_mean)
		# push!(plots2, vae_sample)
		
		# IWAE: sample 1 z and 1 x
		iwae_mean, iwae_sample = iwae_mean_smaple_images(x)
		push!(plots2, iwae_mean)
		push!(plots2, iwae_sample)
	end

	# plot the plots in a 4 x 5 grid
	plot(plots2 ..., layout = (4, 3), size=(600, 600))
end

# ╔═╡ cf33c5fe-819c-4efd-a8ed-f2e99b40bf8f
savefig("IWAE_visualization_k=5.png")

# ╔═╡ Cell order:
# ╠═e5416a10-a01c-11eb-1cab-793d2df3d6c1
# ╠═2f9e8ec8-dcf7-4e8c-b7ac-2a15f90fc60c
# ╠═7d758b6c-9cfd-4085-bd84-230e91f8190b
# ╠═1eb02d21-8125-4955-9cd6-6c0a0ac4b586
# ╠═5f3207d1-b95d-4038-b708-1ceb421cca5d
# ╠═3fad4b3c-d4ea-443c-b64d-18b9e127d3cf
# ╟─ce104a6c-a383-4d04-a81e-45204f873ac2
# ╠═f9f90fdf-c8a3-4478-b9ca-76904c3b1d91
# ╟─e4951f61-99ee-4295-ae51-c0deffd3c66a
# ╠═c11be4e5-644d-41a2-9672-4582bfb771d4
# ╠═59e7823a-1a15-4b4a-8675-7bb2ffaa3dc6
# ╠═df7731fe-81c3-4b7d-b782-35ff8746adf9
# ╠═cf3ebd63-c0f8-4478-a132-6b7a95897ab2
# ╠═e5137cdc-246b-474d-8b52-cb1bf7eb1c24
# ╠═deaad4a8-c5af-4a71-833f-735d2ef6d11d
# ╠═9a5fa88c-65c6-472a-9955-75be909319e0
# ╠═f87dd3c5-d4cd-4d0f-b317-3f56cbaaa61d
# ╠═5611074c-082e-42fa-91de-3ac6553b0160
# ╠═e1b6014d-0425-48d1-9e3e-841fe7fbfa7f
# ╠═2bde411e-db3b-42e5-aa6a-909c430d0c6c
# ╠═a1ead5c7-8a2e-4eda-ace2-fe44c9f55c19
# ╠═78a21a4f-925c-4029-a7e2-441f74aaf407
# ╟─b353df2e-1197-45da-b44e-22133be729e3
# ╠═98af9656-8fd7-450c-af9c-f97ec13156dd
# ╠═0fb26a86-d1aa-4b15-a3c2-681ba08343b9
# ╟─d8b579ee-575c-49bf-aafb-544347719a9c
# ╠═88118316-1b8f-4cd9-a8b7-0f8d925205dc
# ╠═c07f07fc-642a-47e8-9ec8-4abda3d3a34d
# ╠═647b08cf-0853-4c90-b33e-2b04548c5b2b
# ╠═64292978-40b6-4cef-ab62-768bfbf4854e
# ╠═a4ca2198-5bab-4fae-ae14-ccb4d17c3186
# ╠═7b949260-94e3-47ac-96b9-66fc73dfa234
# ╟─47d5146b-bd02-4347-8701-9ed68e3d3405
# ╠═958a7118-7283-4509-b1cd-30d2458e2489
# ╠═7c675eeb-0b74-4346-9776-fbfbf72b7b8e
# ╠═137d48da-a8a2-43e2-94ea-36771aa4d9fa
# ╠═d7f41eb6-48d4-47f4-8588-25f3accaf381
# ╠═cf33c5fe-819c-4efd-a8ed-f2e99b40bf8f
