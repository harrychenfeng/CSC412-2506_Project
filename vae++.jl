### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ d402633e-8c18-11eb-119d-017ad87927b0
using Flux

# ╔═╡ c70eaa72-90ad-11eb-3600-016807d53697
using StatsFuns: log1pexp #log(1 + exp(x))

# ╔═╡ a42a19d6-c61e-465f-a3cc-4e9dc7eff826
using BSON: @save

# ╔═╡ 15de28c7-576f-4f80-9b57-f1112ce14cbf
using BSON

# ╔═╡ 0a761dc4-90bb-11eb-1f6c-fba559ed5f66
using Plots ##

# ╔═╡ 5344b278-82b1-4a14-bf9e-350685c6d57e
using Images

# ╔═╡ 0155586a-9ffc-4319-bb93-6aeab31f670e
using Statistics

# ╔═╡ 5a62989d-6b26-4209-a7c2-fde82d5a87b2
using ConditionalDists

# ╔═╡ a6d94aa9-e4af-4dd1-a31b-b07e53a11e17
using Flux: onehotbatch

# ╔═╡ 63b0f25f-076c-4b0f-b09b-5ecd9d317e13
using Distributions

# ╔═╡ ebfce7a9-6f8e-4977-91cc-50fb4f27c395
using SpecialFunctions

# ╔═╡ 54749c92-8c1d-11eb-2a54-a1ae0b1dc587
# load the original greyscale digits
train_digits = Flux.Data.MNIST.images(:train)

# ╔═╡ 176f0938-8c1e-11eb-1135-a5db6781404d
# convert from tuple of (28,28) digits to vector (784,N) 
greyscale_MNIST = hcat(float.(reshape.(train_digits,:))...)

# ╔═╡ c6fa2a9c-8c1e-11eb-3e3c-9f8f5c218dec
# binarize digits
binarized_MNIST = greyscale_MNIST .> 0.5

# ╔═╡ 9e7e46b0-8e84-11eb-1648-0f033e4e6068
# partition the data into batches of size BS
BS = 200

# ╔═╡ 743d473c-8c1f-11eb-396d-c92cacb0235b
# batch the data into minibatches of size BS
batches = Flux.Data.DataLoader(binarized_MNIST, batchsize=BS)

# ╔═╡ db655546-8e84-11eb-21df-25f7c8e82362
# confirm dimensions are as expected (D,BS)
size(first(batches))

# ╔═╡ 45bc7e00-90ac-11eb-2d62-092a13dd1360
md"""
### Bernoulli Log Density

The Bernoulli distribution $\text{Ber}(x \mid \mu)$ where $\mu \in [0,1]$ is difficult to optimize for a few reasons. One solution is to parameterize the "logit-means": $y = \log(\frac{\mu}{1-\mu})$.

We can exploit further numerical stability, e.g. in computing $\log(1 + exp(x))$, using library provided functions `log1pexp`
"""


# ╔═╡ e12a5b5e-90ad-11eb-25a8-43c9aff1e0db
# Numerically stable bernoulli density, why do we do this?
function bernoulli_log_density(x, logit_means)
  """Numerically stable log_likelihood under bernoulli by accepting μ/(1-μ)"""
	b = x .* 2 .- 1 # [0,1] -> [-1,1]
  	return - log1pexp.(-b .* logit_means)
end

# ╔═╡ 3b07d20a-8e88-11eb-1956-ddbaaf178cb3
md"""
## Model Implementation

- `log_prior` that computes the log-density of a latent representation under the prior distribution.
- `decoder` that takes a latent representation $z$ and produces a 784-dimensional vector $y$. This will be a simple neural network with the following architecture: a fully connected layer with 500 hidden units and `tanh` non-linearity, a fully connected output layer with 784-dimensions. The output will be unconstrained, no activation function.
- `log_likelihood` that given an array binary pixels $x$ and the output from the decoder, $y$ corresponding to "logit-means" of the pixel Bernoullis $y = log(\frac{\mu}{1-\mu})$ compute the **log-**likelihood under our model. 
- `joint_log_density` that uses the `log_prior` and `log_likelihood` and gives the log-density of their joint distribution under our model $\log p_\theta(x,z)$.

Note that these functions should accept a batch of digits and representations, an array with elements concatenated along the last dimension.
"""

# ╔═╡ ad5b81a7-115d-4c8f-852d-298dbbd81c4b
function factorized_gaussian_log_density(samples, μ, logσ)
	σ = exp.(logσ)
	return sum(-0.5*((samples.-μ)./σ).^2 .- log.(σ.*sqrt(2π)),dims=1)
end

# ╔═╡ ce50c994-90af-11eb-3fc1-a3eea9cda1a2
log_prior(z) = factorized_gaussian_log_density(z, 0, 0)

# ╔═╡ 3b386e56-90ac-11eb-31c2-29ba365a6967
Dz, Dh, Ddata = 2, 500, 28^2

# ╔═╡ d7415d20-90af-11eb-266b-b3ea86750c98
decoder = Chain(Dense(Dz, Dh, tanh), Dense(Dh, Ddata)) # You can use Flux's Chain and Dense here

# ╔═╡ 5b8721dc-8ea3-11eb-3ace-0b13c00ce256
function log_likelihood(x,z)
  """ Compute log likelihood log_p(x|z)"""
	# use numerically stable bernoulli
	return sum(bernoulli_log_density(x, decoder(z)),dims=1)
end

# ╔═╡ 0afbe054-90b0-11eb-0233-6faede537bc4
joint_log_density(x,z) = log_prior(z) .+ log_likelihood(x,z)

# ╔═╡ b8a20c8c-8ea4-11eb-0d48-a37047ab70c5
md"""
## Amortized Approximate Inference with Learned Variational Distribution

Now that we have set up a model, we would like to learn the model parameters $\theta$.
Notice that the only indication for *how* our model should represent digits in $z \in \mathbb{R}^2$ is that they should look like our prior $\mathcal{N}(0,1)$.

How should our model learn to represent digits by 2D latent codes? 
We want to maximize the likelihood of the data under our model $p_\theta(x) = \int p_\theta(x,z) dz = \int p_\theta(x \mid z)p(z) dz$.

We have learned a few techniques to approximate these integrals, such as sampling via MCMC. 
Also, 2D is a low enough latent dimension, we could numerically integrate, e.g. with a quadrature.

Instead, we will use variational inference and find an approximation $q_\phi(z) \approx p_\theta(z \mid x)$. This approximation will allow us to efficiently estimate our objective, the data likelihood under our model. Further, we will be able to use this estimate to update our model parameters via gradient optimization.

Following the motivating paper, we will define our variational distribution as $q_\phi$ also using a neural network. The variational parameters, $\phi$ are the weights and biases of this "encoder" network.

This encoder network $q_\phi$ will take an element of the data $x$ and give a variational distribution over latent representations. In our case we will assume this output variational distribution is a fully-factorized Gaussian.
So our network should output the $(\mu, \log \sigma)$.

To train our model parameters $\theta$ we will need also train variational parameters $\phi$.
We can do both of these optimization tasks at once, propagating gradients of the loss to update both sets of parameters.

The loss, in this case, no longer being the data likelihood, but the Evidence Lower BOund (ELBO).

1. Implement `log_q` that accepts a representation $z$ and parameters $\mu, \log \sigma$ and computes the logdensity under our variational family of fully factorized guassians.
1. Implement `encoder` that accepts input in data domain $x$ and outputs parameters to a fully-factorized guassian $\mu, \log \sigma$. This will be a neural network with fully-connected architecture, a single hidden layer with 500 units and `tanh` nonlinearity and fully-connected output layer to the parameter space.
2. Implement `elbo` which computes an unbiased estimate of the Evidence Lower BOund (using simple monte carlo and the variational distribution). This function should take the model $p_\theta$, the variational model $q_\phi$, and a batch of inputs $x$ and return a single scalar averaging the ELBO estimates over the entire batch.
4. Implement simple loss function `loss` that we can use to optimize the parameters $\theta$ and $\phi$ with `gradient`. We want to maximize the lower bound, with gradient descent. (This is already implemented)

"""

# ╔═╡ 615e59c6-90b6-11eb-2598-d32538e14e8f
log_q(z, q_μ, q_logσ) = factorized_gaussian_log_density(z, q_μ, q_logσ)

# ╔═╡ c0d81a4f-4f71-480e-b9eb-610c6dc3cf78
function unpack_guassian_params(output)
	μ, logσ = output[1:2,:], output[3:4,:]
	return μ, logσ
end

# ╔═╡ 95c2435e-5c0a-479b-9def-d9b6459368e7
encoder = Chain(Dense(Ddata, Dh, tanh), Dense(Dh, Dz*2), unpack_guassian_params)

# ╔═╡ 16affe57-ab45-406d-8759-07786ca1765b
sample_from_var_dist(μ, logσ) = (randn(size(μ)) .* exp.(logσ) .+ μ)

# ╔═╡ ccf226b8-90b6-11eb-15a2-d30c9e27aebb
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

# ╔═╡ f00b5444-90b6-11eb-1e0d-5d034735ec0e
function loss(x)
  return -elbo(x)
end

# ╔═╡ 70ccd9a4-90b7-11eb-1fb2-3f7aff4073a0
md"""
## Optimize the model and amortized variational parameters

If the above are implemented correctly, stable numerically, and differentiable automatically then we can train both the `encoder` and `decoder` networks with graident optimzation.

We can compute `gradient`s of our `loss` with respect to the `encoder` and `decoder` parameters `theta` and `phi`.

We can use a `Flux.Optimise` provided optimizer such as `ADAM` or our own implementation of gradient descent to `update!` the model and variational parameters.

Use the training data to learn the model and variational networks.
"""

# ╔═╡ 5efb0baa-90b8-11eb-304f-7dbb8d5c0ba6
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

# ╔═╡ c86a877c-90b9-11eb-31d8-bbcb71e4fa66
train!(encoder, decoder, batches, nepochs=3)

# ╔═╡ beec9c0c-d5cb-41fa-a5b6-7ba5da4afb68
begin
	@save "encoder.bson" encoder 
	@save "decoder.bson" decoder 
end

# ╔═╡ bd5408bc-9998-4bf7-9752-5823f79354f8
begin
	BSON.load("encoder.bson", @__MODULE__)
	BSON.load("decoder.bson", @__MODULE__)
end

# ╔═╡ 0fbe4662-7777-4a23-a26c-c3f8968414b9
q_μ, q_logσ = encoder(first(batches))

# ╔═╡ 17c5ddda-90ba-11eb-1fce-93b8306264fb
md"""
## Visualizing the Model Learned Representation

We will use the model and variational networks to visualize the latent representations of our data learned by the model.

We will use a variatety of qualitative techniques to get a sense for our model by generating distributions over our data, sampling from them, and interpolating in the latent space.
"""

# ╔═╡ 2029fc15-3ed9-4f99-b585-c93fdcdc66fb
function calculate_bernoulli_mean(logit_means)
	return exp.(logit_means) ./ (1 .+ exp.(logit_means))
end

# ╔═╡ de41eda0-2636-4f87-b791-286a84f744ff
md"""
### Larger Latent Space
###### Experimented a 3D latent space and make visualization
"""

# ╔═╡ 0111f7f1-90c8-4ae7-ad75-68b155d4bd30
Dz_3d = 3

# ╔═╡ 7a86763d-fa11-48ac-94f0-7bd9874af9c5
decoder_3d = Chain(Dense(Dz_3d, Dh, tanh), Dense(Dh, Ddata))

# ╔═╡ 3553b2af-d10b-4df5-b67a-ba3d252b7e0e
function log_likelihood_larger(x,z)
  """ Compute log likelihood log_p(x|z)"""
	return sum(bernoulli_log_density(x, decoder_3d(z)),dims=1)
end

# ╔═╡ adc7cc1a-dd3d-48bc-8c34-7d30b684199d
function unpack_guassian_params_3d(output)
	μ, logσ = output[1:3,:], output[4:6,:]
	return μ, logσ
end

# ╔═╡ 65718db9-5ab1-46d0-9dfb-3b234c956e1f
encoder_3d = Chain(Dense(Ddata, Dh, tanh), Dense(Dh, Dz_3d*2), unpack_guassian_params_3d)

# ╔═╡ 28172ef1-06c6-4236-b027-b55c79ce94f1
sample_from_var_dist_3d(μ, logσ) = (randn(size(μ)) .* exp.(logσ) .+ μ)

# ╔═╡ fdd02429-60d1-4c65-bc0d-2ea121d1a712
joint_log_density_3d(x,z) = log_prior(z) .+ log_likelihood_larger(x,z)

# ╔═╡ 119a6e70-d698-489e-9605-1757b8429f57
function elbo_3d(x)
	q_μ, q_logσ = encoder_3d(x)
  	z = sample_from_var_dist_3d(q_μ, q_logσ)
  	joint_ll = joint_log_density_3d(x,z)
  	log_q_z = log_q(z, q_μ, q_logσ)
  	elbo_estimate = sum(joint_ll - log_q_z)/size(x)[2]
	return elbo_estimate, q_logσ
end

# ╔═╡ 1f3e4948-b75a-4d78-875f-0f8586e49e82
function loss_3d(x)
  elbo_estimate, logσ = elbo_3d(x)
  return -elbo_estimate, logσ
end

# ╔═╡ c6d56e0f-0284-4c74-b632-3ae1cd1cae6a
# logσs = Any[]

# ╔═╡ 0a731566-69e6-409a-bb9e-9dc371dfb890
function train_3d!(enc, dec, data; nepochs=100)
	params = Flux.params(enc, dec)
	opt = ADAM()
	@info "Begin training in 3D latent space"
	for epoch in 1:nepochs
		b_loss = 0
		logσ = Any[]
		for batch in data
			grads = Flux.gradient(params) do
				b_loss, logσ = loss_3d(batch)
				# push!(logσs, logσ)
				# if epoch == nepochs
				# 	var = (exp.(logσ)) .^ 2
				# 	vs = size(var)
				# end
				return b_loss
			end
			
			Flux.Optimise.update!(opt, params, grads)
		end
		var = (exp.(logσ)) .^ 2
		# var_size = size(var)
		# @info "vs: $var_size"
		scatter(var[1,:], var[2,:], var[3,:])
		# Optional: log loss using @info "Epoch $epoch: loss:..."
		@info "Epoch $epoch: loss:$b_loss"
		# Optional: visualize training progress with plot of loss
		# Optional: save trained parameters to avoid retraining later
	end
	@info "Training in 3D is done"
	# return nothing, this mutates the parameters of enc and dec!
	# return logσs
end

# ╔═╡ d6f3c660-c632-4795-8bb9-35e456bebac4
train_3d!(encoder_3d, decoder_3d, batches, nepochs=3)

# ╔═╡ f5d2e08b-8975-4190-a033-e6630dea3ab9
begin
	@save "encoder_3d.bson" encoder_3d 
	@save "decoder_3d.bson" decoder_3d
end

# ╔═╡ adc743b0-4174-4fa5-9c11-b30817cc3768
begin
	BSON.load("encoder_3d.bson", @__MODULE__)
	BSON.load("decoder_3d.bson", @__MODULE__)
end

# ╔═╡ b6a2b5d8-5258-4b75-8611-d25a4a075753
q_μ_3d, q_logσ_3d = encoder_3d(first(batches))

# ╔═╡ e2c07c18-43c8-4899-b083-05c8180dc7d3
# Training set labels
begin
	train_labels = Flux.Data.MNIST.labels(:train)
	label_batches = Flux.Data.DataLoader(train_labels, batchsize=BS)
	labels = first(label_batches)
end

# ╔═╡ 78714227-094e-4a00-a72c-accfb50e1b71
scatter(q_μ_3d[1,:], q_μ_3d[2,:], q_μ_3d[3,:], group=labels, title="A batch 3D latent space of mean vectors for μ", xlabel="z1 for mean μ", ylabel="z2 for mean μ", zlabel="z3 for mean μ")

# ╔═╡ 5ca39a1b-4871-4f48-9934-99c88fb504ba
md"""
###### Comparison with baselines
"""

# ╔═╡ 79015407-a145-44ed-853d-ca7c89676ddc
# Helper function for drawing the MNIST digit in 28*28 shape
function draw_image(x)
	dim = ndims(x)
	if dim == 2
		x_2d = reshape(x, 28, 28, :)
		return Gray.(x_2d)
	else
		x_3d = reshape(x, 28, 28)
		return Gray.(x_3d)
	end
end

# ╔═╡ b7ae23ec-1612-40b9-8020-21d5ee5f4c48
md"""
Baseline (2D latent space)
"""

# ╔═╡ b10f4fa4-7fcb-4bc1-975a-55c8079a616a
begin
	# 1. Sample 10 2D z from the prior p(z)
	zs = Any[]
	for i in 1:10
		sample_z = randn(2,)
		push!(zs, sample_z)
	end
end

# ╔═╡ e4b267fb-c2e9-4341-b261-860b12648127
begin
	# 2. decode each z to get logit-means
	plots1, plots2, plots3 = Any[], Any[], Any[]
	plots = Any[]
	for i in 1:10
		logit_means = decoder(zs[i])
		# 3. Transfer logit-means to Bernoulli means μ
		bern_mean = calculate_bernoulli_mean(logit_means)
		push!(plots, draw_image(bern_mean))
		# 5. Sample 3 examples from Bernoulli
		samples1 = rand(Float64, size(bern_mean)) .< bern_mean
		push!(plots1, draw_image(samples1))
		samples2 = rand(Float64, size(bern_mean)) .< bern_mean
		push!(plots2, draw_image(samples2))
		samples3 = rand(Float64, size(bern_mean)) .< bern_mean
		push!(plots3, draw_image(samples3))
	end
end

# ╔═╡ 670121db-0989-4939-86e6-888194affb41
md"""
Model with larger dimension (3D latent space)
"""

# ╔═╡ 92b6e763-522b-444e-9127-9ff51ef1239b
begin
	# 1. Sample 10 3D z from the prior p(z)
	zs_3d = Any[]
	for i in 1:10
		sample_z_3d = randn(3,)
		push!(zs_3d, sample_z_3d)
	end
end

# ╔═╡ a08ff202-3e7f-413e-8801-91d609409753
begin
	# 2. decode each z to get logit-means
	plots1_3d, plots2_3d, plots3_3d = Any[], Any[], Any[]
	plots_3d = Any[]
	for i in 1:10
		logit_means_3d = decoder_3d(zs_3d[i])
		# 3. Transfer logit-means to Bernoulli means μ
		bern_mean_3d = calculate_bernoulli_mean(logit_means_3d)
		push!(plots_3d, draw_image(bern_mean_3d))
		# 5. Sample 3 examples from Bernoulli 
		samples1_3d = rand(Float64, size(bern_mean_3d)) .< bern_mean_3d
		push!(plots1_3d, draw_image(samples1_3d))
		samples2_3d =  rand(Float64, size(bern_mean_3d)) .< bern_mean_3d
		push!(plots2_3d, draw_image(samples2_3d))
		samples3_3d = rand(Float64, size(bern_mean_3d)) .< bern_mean_3d
		push!(plots3_3d, draw_image(samples3_3d))
	end
end

# ╔═╡ 9a5d4ff2-9ccb-4883-84fe-af3f3d52c568
begin
	# 6. Display all plots in a single 10 x 4 grid
	p_3d = plot(layout = (10,1), size=(500,1200))
	for i in 1:10
		heatmap!(cat(plots[i], plots1[i], plots2[i], plots3[i], plots_3d[i], plots1_3d[i], plots2_3d[i], plots3_3d[i],dims=2), subplot=i)
	end
	plot(p_3d)
end

# ╔═╡ 2a3caf33-8f54-42a0-ad52-a1fae4863a6a
md"""
###### Variance respond as the dimensionality of latent space increases
2D latent space v.s. 3D latent space
"""

# ╔═╡ 7edbeb65-ed32-4cab-ab20-37e65c820334
q_var_3d = (exp.(q_logσ_3d)).^2

# ╔═╡ 4759682f-9239-42c9-9a30-d70dc3939285
q_var_2d = (exp.(q_logσ)).^2

# ╔═╡ 848c3d78-f8c6-4f44-96b2-e8f1406a56c5
begin
	plot(q_var_3d[1,:], label="3D z1", lw = 1.5)
	plot!(q_var_3d[2,:], label="3D z2", lw = 1.5)
	plot!(q_var_3d[3,:], label="3D z3", lw = 1.5)
	xlabel!("Number of samples")
	ylabel!("Variance")
	title!("Variance along each dimension in 3D latent space")
end

# ╔═╡ 4936599e-d2f5-4901-b049-6ae55cdaca24
mean(q_var_3d, dims=2)

# ╔═╡ 9054f4f9-6a31-404a-bc7e-6be1480e00c6
begin
	plot(q_var_2d[1,:], label="2D z1", lw = 1.5)
	plot!(q_var_2d[2,:], label="2D z2", lw = 1.5)
	xlabel!("Number of samples")
	ylabel!("Variance")
	title!("Variance along each dimension in 2D latent space")
end

# ╔═╡ 37d20d35-72b9-496e-9dd0-a81a3d5c69cd
mean(q_var_2d, dims=2)

# ╔═╡ 8f446c4a-6df9-45d1-806a-3433a039229d
md"""
As the dimension of latent space increases, the variance along each dimension tends to increase.
"""

# ╔═╡ f0f3347b-a89f-4ed3-94b6-1be2482fe954
md"""
###### Variance respond as the dimensionality of latent space increases
TODO: Plot variance for 2D latent space v.s. 3D latent space during training
"""

# ╔═╡ 1b1bd03a-09a7-49f9-96e7-45aaa35ef8f4
md"""
### Condition on MNIST Digit Supervision
"""

# ╔═╡ 772f74d7-097b-40e6-a8c9-ee536dcffadc
md"""
Horizontally concate labels to data
"""

# ╔═╡ 2c8ae3f0-9e01-40b3-9cb0-0d651a048662
encoder_cond = Chain(Dense(Ddata+10, Dh, tanh), Dense(Dh, Dz_3d*2), unpack_guassian_params_3d)

# ╔═╡ 5926c5c0-c552-4802-8447-58e526e00659
decoder_cond = Chain(Dense(Dz_3d, Dh, tanh), Dense(Dh, Ddata+10))

# ╔═╡ f19287fa-1ed4-4994-b59a-2731f8accec0
md"""
Change labels to one-hot encoding vectors
"""

# ╔═╡ 89af27db-3cc4-424d-9741-473e863a95e4
onehot_labels = onehotbatch(train_labels, [:0, :1, :2, :3, :4, :5, :6, :7, :8, :9])

# ╔═╡ debff640-513f-4b1c-bddf-0cbbe33b0522
batches_cond = Flux.Data.DataLoader(cat(binarized_MNIST, onehot_labels, dims=1), batchsize=BS)

# ╔═╡ 2191de84-b685-4b25-816e-f38a6e12db3d
function log_likelihood_cond(x,z)
  """ Compute log likelihood log_p(x|z)"""
	# use numerically stable bernoulli
	return sum(bernoulli_log_density(x, decoder_cond(z)),dims=1)
end

# ╔═╡ 85186d7a-2f4b-4766-8bcf-8edb415af4f8
joint_log_density_cond(x,z) = log_prior(z) .+ log_likelihood_cond(x,z)

# ╔═╡ 296bea20-2efa-4a1f-bc14-7d3a61e45bd2
function elbo_3d_cond(x)	
	q_μ, q_logσ = encoder_cond(x)
  	z = sample_from_var_dist_3d(q_μ, q_logσ)
  	joint_ll = joint_log_density_cond(x,z)
  	log_q_z = log_q(z, q_μ, q_logσ)
  	elbo_estimate = sum(joint_ll - log_q_z)/size(x)[2]
  	return elbo_estimate
end

# ╔═╡ 2baf4e94-8651-494b-8c76-b05fdba16c99
function loss_3d_cond(x)
  	return -elbo_3d_cond(x)
end

# ╔═╡ f7e9c8c1-7939-4551-9267-28e58bb63191
function train_cond!(enc, dec, data; nepochs=100)
	params = Flux.params(enc, dec)
	opt = ADAM()
	@info "Begin training in 3D latent space with given labels"
	for epoch in 1:nepochs
		b_loss = 0
		for batch in data
			grads = Flux.gradient(params) do
				b_loss = loss_3d_cond(batch)
				return b_loss
			end
			Flux.Optimise.update!(opt, params, grads)
		end
		@info "Epoch $epoch: loss:$b_loss"
	end
	@info "Training in 3D latent space(labels) is done"
end

# ╔═╡ 780d91c4-c606-48d5-bda6-7512f4ab2d82
train_cond!(encoder_cond, decoder_cond, batches_cond, nepochs=3)

# ╔═╡ d41c401c-3ac9-47fa-8f88-c6e81f5d092d
begin
	@save "encoder_3d_labels.bson" encoder_cond 
	@save "decoder_3d_labels.bson" decoder_cond
end

# ╔═╡ c2ef34f8-2ff9-4731-aed3-0262d4c0b733
begin
	BSON.load("encoder_3d_labels.bson", @__MODULE__)
	BSON.load("decoder_3d_labels.bson", @__MODULE__)
end

# ╔═╡ 0695dd02-e617-4110-9f5e-234f5cfcad31
q_μ_cond, q_logσ_cond = encoder_cond(first(batches_cond))

# ╔═╡ 395bec09-3908-4095-afdb-ade87f6b6078
md"""
###### Visualize latent representation
"""

# ╔═╡ 1ad7f796-e1f4-4bba-8e38-3f1ba5628474
scatter(q_μ_cond[1,:], q_μ_cond[2,:], q_μ_cond[3,:], group=labels, title="A batch 3D latent space of mean vectors for μ given labels", xlabel="z1 for mean μ", ylabel="z2 for mean μ", zlabel="z3 for mean μ")

# ╔═╡ bc46e68f-ea1d-4f3a-b044-38bd7107bd6d
begin
	# 1. Sample 10 3D z from the prior p(z)
	zs_cond = Any[]
	for i in 1:10
		sample_z_cond = randn(3,)
		push!(zs_cond, sample_z_cond)
	end
end

# ╔═╡ 6189ad86-5539-4ebd-9413-ad3592b24d24
# Helper function for drawing the MNIST digit in 28*28 shape
function draw_image_cond(x)
	dim = ndims(x)
	if dim == 2
		x_2d = reshape(x, 28, 28, :)
		return Gray.(x_2d)
	else
		x_3d = reshape(x, 28, 28)
		return Gray.(x_3d)
	end
end

# ╔═╡ 5902256e-c3e6-4194-b141-eb83e7171206
begin
	# 2. decode each z to get logit-means
	plots1_cond = Any[]
	plots_cond = Any[]
	for i in 1:10
		logit_means_cond = decoder_cond(zs_cond[i])
		# 3. Transfer logit-means to Bernoulli means μ
		bern_mean_cond = vec(calculate_bernoulli_mean(logit_means_cond))[1:784]
		# bs = size(bern_mean_cond)
		# @info "bern size: $bs"
		push!(plots_cond, draw_image_cond(bern_mean_cond))
		# 5. Sample 1 example from Bernoulli 
		samples1_cond = rand(Float64, size(bern_mean_cond)) .< bern_mean_cond
		push!(plots1_cond, draw_image_cond(samples1_cond))
		
	end
end

# ╔═╡ 54c2cccd-2f3d-4bc0-a7eb-4f00367ad619
begin
	# 6. Display all plots in a single 10 x 4 grid
	p_cond = plot(layout = (10,1), size=(500,1200))
	for i in 1:10
		heatmap!(cat(plots_cond[i], plots1_cond[i],dims=2), subplot=i)
	end
	plot(p_cond)
end

# ╔═╡ f50eeab6-7cda-4593-9f3b-6ac6dc258ba6
md"""
###### Semi-supervised learning
"""

# ╔═╡ 5836cc48-dc6b-4768-bcb4-b7a5693e3deb
begin
	onehot_semi = Matrix(onehot_labels)
	index = rand(1:60000,45000)
	onehot_semi[:,index] = zeros(10, 45000)
end

# ╔═╡ 97bff68e-54af-4b4f-8d0f-5fcda48be845
batches_semi = Flux.Data.DataLoader(cat(binarized_MNIST, onehot_semi, dims=1), batchsize=BS)

# ╔═╡ f248ae66-6a8f-47d8-808a-8b5b3efaf795
encoder_semi = Chain(Dense(Ddata+10, Dh, tanh), Dense(Dh, Dz_3d*2), unpack_guassian_params_3d)

# ╔═╡ d93b609a-4a5b-4c5f-af43-801083d8b67f
decoder_semi = Chain(Dense(Dz_3d, Dh, tanh), Dense(Dh, Ddata+10))

# ╔═╡ bb620ae9-7a38-4c39-8026-fb3a9ffb768e
function train_semi!(enc, dec, data; nepochs=100)
	params = Flux.params(enc, dec)
	opt = ADAM()
	@info "Begin training in 3D latent space with given semi labels"
	for epoch in 1:nepochs
		b_loss = 0
		for batch in data
			grads = Flux.gradient(params) do
				b_loss = loss_3d_cond(batch)
				return b_loss
			end
			Flux.Optimise.update!(opt, params, grads)
		end
		@info "Epoch $epoch: loss:$b_loss"
	end
	@info "Training in 3D latent space(semi labels) is done"
end

# ╔═╡ beade4cf-6b98-48d5-ad02-8102360fa55a
train_cond!(encoder_semi, decoder_semi, batches_semi, nepochs=3)

# ╔═╡ 024ea208-3ae1-43e8-8758-d049d0eff5d3
begin
	@save "encoder_semi.bson" encoder_semi
	@save "decoder_semi.bson" decoder_semi
end

# ╔═╡ 1faa33a5-b679-442a-b983-16cab2b48ad9
begin
	BSON.load("encoder_semi.bson", @__MODULE__)
	BSON.load("decoder_semi.bson", @__MODULE__)
end

# ╔═╡ 0fc0995f-f87a-4c13-b33a-ab01f86f3d1c
begin
	# 2. decode each z to get logit-means
	plots1_semi = Any[]
	plots_semi = Any[]
	for i in 1:10
		logit_means_semi = decoder_cond(zs_cond[i])
		# 3. Transfer logit-means to Bernoulli means μ
		bern_mean_semi = vec(calculate_bernoulli_mean(logit_means_semi))[1:784]
		# bs = size(bern_mean_cond)
		# @info "bern size: $bs"
		push!(plots_semi, draw_image_cond(bern_mean_semi))
		# 5. Sample 1 example from Bernoulli 
		samples1_semi = rand(Float64, size(bern_mean_semi)) .< bern_mean_semi
		push!(plots1_semi, draw_image_cond(samples1_semi))
		
	end
end

# ╔═╡ 3639bf58-809f-4221-8c91-8abbe218d0b4
begin
	# 6. Display all plots in a single 10 x 4 grid
	p_semi = plot(layout = (10,1), size=(500,1200))
	for i in 1:10
		heatmap!(cat(plots_semi[i], plots1_semi[i],dims=2), subplot=i)
	end
	plot(p_semi)
end

# ╔═╡ 5bd4471f-d3c3-451d-94ee-05a30132e2b9
md"""
### Optimizing Different Divergences
"""

# ╔═╡ 077026fc-808f-46ef-8308-d32f50b0e12f
function create_enc_dec()
	encoder = Chain(Dense(Ddata, Dh, tanh), Dense(Dh, Dz*2), unpack_guassian_params)
	decoder = Chain(Dense(Dz, Dh, tanh), Dense(Dh, Ddata))
	return encoder, decoder
end

# ╔═╡ 1ebf3383-03c8-434d-8f09-9621344c686b
enc1, dec1 = create_enc_dec()

# ╔═╡ 2bd2d7da-5798-4104-89cf-a81f4d553be5
function elbo_js(x)	
	q_μ, q_logσ = enc1(x)
  	z = sample_from_var_dist(q_μ, q_logσ)
  	joint_ll = joint_log_density(x,z)
  	log_q_z = log_q(z, q_μ, q_logσ)
	p = exp.(joint_ll)
	q = exp.(log_q_z)
	# kl(p,q) = sum(p .* log.(p ./ q))
	m = 0.5*log.(p+q)
	# pm = kl(p,m)
	# qm = kl(q,m)
	pm = sum(joint_ll - m)/size(x)[2]
	qm = sum(log_q_z - m)/size(x)[2]
	# elbo_estimate = sum(p .* log.(p ./ q)- p + q)/size(x)[2]
  	# elbo_estimate = sum(joint_ll - log_q_z)/size(x)[2]
	elbo_estimate = 0.5*pm + 0.5*qm
  	return elbo_estimate
end

# ╔═╡ 0d27ad7a-d3a2-4873-bf5f-f6585fc63882
function loss_js(x)
  	return -elbo_js(x)
end

# ╔═╡ 1c1dd8a7-697e-45e2-9734-f221b2e46978
function train_js!(enc, dec, data; nepochs=100)
	params = Flux.params(enc, dec)
	opt = ADAM()
	@info "Begin training in 2D latent space using JS Divergence"
	for epoch in 1:nepochs
		b_loss = 0
		for batch in data
			# compute gradient wrt loss
			grads = Flux.gradient(params) do
				b_loss = loss_js(batch)
				return b_loss
			end
			# update parameters
			Flux.Optimise.update!(opt, params, grads)
		end
		@info "Epoch $epoch: loss:$b_loss"
	end
	@info "Training in 2D using JS Divergence is done"
end

# ╔═╡ 53eeb659-3083-45ea-a001-23e16c29388a
train_js!(enc1, dec1, batches, nepochs=1)

# ╔═╡ 48d0afee-667b-4b9f-8610-2abb5973ef86
# begin
# 	@save "encoder_js.bson" encoder_js
# 	@save "decoder_js.bson" decoder_js
# end

# ╔═╡ 8b190d6d-ed7e-4e19-9228-6ec3a8d6934e
# begin
# 	BSON.load("encoder_js.bson", @__MODULE__)
# 	BSON.load("decoder_js.bson", @__MODULE__)
# end

# ╔═╡ 52a9a59f-1e3a-484f-87bd-22dbdc0d704b
function visualize_samples(decoder)
	plots1 = Any[]
	plots = Any[]
	for i in 1:10
		# 1. Sample 10 3D z from the prior p(z)
		z = randn(2,)
		# 2. decode each z to get logit-means
		logit_means = decoder(z)
		# 3. Transfer logit-means to Bernoulli means μ
		bern_mean = calculate_bernoulli_mean(logit_means)
		push!(plots, draw_image(bern_mean))
		# 5. Sample 1 example from Bernoulli 
		samples1 = rand(Float64, size(bern_mean)) .< bern_mean
		push!(plots1, draw_image(samples1))
	end
	return plots1, plots
end

# ╔═╡ d25834fb-b564-4a54-aacb-0bac51b45e70
plots1_1, plots_1 = visualize_samples(dec1)

# ╔═╡ aab34582-8cbb-4ff2-ab1b-d660755acbec
function plot_mnist_image(plots, plots1)
	# 6. Display all plots in a single 10 x 4 grid
	p = plot(layout = (10,1), size=(500,1200))
	for i in 1:10
		heatmap!(cat(plots[i], plots1[i],dims=2), subplot=i)
	end
	plot(p)
end

# ╔═╡ 30772ff5-c234-4e4b-8906-e89719be6435
# plot_mnist_image(plots_1, plots1_1)

# ╔═╡ 888a7310-82fe-40af-b4fd-92577fa46e4d
md"""
### More Expressive Likelihood Model
Use beta likelihood model with $α=2$, $β=2$ on float MNIST
"""

# ╔═╡ 619cbde6-05ec-4f2a-af3d-4bc575d0be49
float_MNIST = convert(Array{Float64}, greyscale_MNIST)

# ╔═╡ e4e852e5-1db7-4a5f-943d-ca24df4af351
function beta_log_density(x, logit_means, α, β) 
	b = x .* 2 .- 1
	B = gamma(α)*gamma(β) / gamma(α+β)
	return - log1pexp.(-b .* logit_means / B)
end

# ╔═╡ 072c7766-f386-417d-94f6-cb135e7c1dc1
encoder_beta, decoder_beta = create_enc_dec()

# ╔═╡ 538d688e-3ee0-43a3-bfb1-bd469c065d8d
function log_likelihood_beta(x, z, α, β)
  """ Compute log likelihood log_p(x|z)"""
	return sum(beta_log_density(x, decoder_beta(z), α, β),dims=1)
end

# ╔═╡ e708275d-1b98-4127-a8a0-b6e8b4adf53b
joint_log_density_beta(x,z, α, β) = log_prior(z) .+ log_likelihood_beta(x,z,α, β)

# ╔═╡ b79ea401-e26d-4d69-96b5-544a83f8dba8
function elbo_beta(x)	
	q_μ, q_logσ = encoder_beta(x)
  	z = sample_from_var_dist(q_μ, q_logσ)
  	joint_ll = joint_log_density_beta(x,z, 2, 2)
  	log_q_z = log_q(z, q_μ, q_logσ)
  	elbo_estimate = sum(joint_ll - log_q_z)/size(x)[2]
  	return elbo_estimate
end

# ╔═╡ b2d83b4d-0075-461d-ae8e-5d38badc48fe
function loss_beta(x)
  	return -elbo_beta(x)
end

# ╔═╡ 9652c66e-b144-4dba-9dfe-65c9d57ff339
function train_beta!(enc, dec, data; nepochs=100)
	params = Flux.params(enc, dec)
	opt = ADAM()
	@info "Begin training in 2D latent space using Beta likelihood on float MNIST"
	for epoch in 1:nepochs
		b_loss = 0
		for batch in data
			# compute gradient wrt loss
			grads = Flux.gradient(params) do
				b_loss = loss_beta(batch)
				return b_loss
			end
			# update parameters
			Flux.Optimise.update!(opt, params, grads)
		end
		@info "Epoch $epoch: loss:$b_loss"
	end
	@info "Training in 2D using Beta likelihood is done"
end

# ╔═╡ 71007aa0-5ab4-4e69-831e-55f661572283
float_batches = Flux.Data.DataLoader(float_MNIST, batchsize=BS)

# ╔═╡ 1bfe2a17-5f4a-45a0-bae5-32fd61f1ccf1
train_beta!(encoder_beta, decoder_beta, float_batches, nepochs=3)

# ╔═╡ c8f3397e-fdf2-4334-98c9-96ea390aa099
plots1_beta, plots_beta = visualize_samples(decoder_beta)

# ╔═╡ 405ce3bc-124d-4815-8732-c967a01f1d66
plot_mnist_image(plots_beta, plots1_beta)

# ╔═╡ Cell order:
# ╠═d402633e-8c18-11eb-119d-017ad87927b0
# ╠═54749c92-8c1d-11eb-2a54-a1ae0b1dc587
# ╠═176f0938-8c1e-11eb-1135-a5db6781404d
# ╠═c6fa2a9c-8c1e-11eb-3e3c-9f8f5c218dec
# ╠═9e7e46b0-8e84-11eb-1648-0f033e4e6068
# ╠═743d473c-8c1f-11eb-396d-c92cacb0235b
# ╠═db655546-8e84-11eb-21df-25f7c8e82362
# ╟─45bc7e00-90ac-11eb-2d62-092a13dd1360
# ╠═c70eaa72-90ad-11eb-3600-016807d53697
# ╠═e12a5b5e-90ad-11eb-25a8-43c9aff1e0db
# ╟─3b07d20a-8e88-11eb-1956-ddbaaf178cb3
# ╠═ad5b81a7-115d-4c8f-852d-298dbbd81c4b
# ╠═ce50c994-90af-11eb-3fc1-a3eea9cda1a2
# ╠═3b386e56-90ac-11eb-31c2-29ba365a6967
# ╠═d7415d20-90af-11eb-266b-b3ea86750c98
# ╠═5b8721dc-8ea3-11eb-3ace-0b13c00ce256
# ╠═0afbe054-90b0-11eb-0233-6faede537bc4
# ╟─b8a20c8c-8ea4-11eb-0d48-a37047ab70c5
# ╠═615e59c6-90b6-11eb-2598-d32538e14e8f
# ╠═c0d81a4f-4f71-480e-b9eb-610c6dc3cf78
# ╠═95c2435e-5c0a-479b-9def-d9b6459368e7
# ╠═16affe57-ab45-406d-8759-07786ca1765b
# ╠═ccf226b8-90b6-11eb-15a2-d30c9e27aebb
# ╠═f00b5444-90b6-11eb-1e0d-5d034735ec0e
# ╟─70ccd9a4-90b7-11eb-1fb2-3f7aff4073a0
# ╠═5efb0baa-90b8-11eb-304f-7dbb8d5c0ba6
# ╠═c86a877c-90b9-11eb-31d8-bbcb71e4fa66
# ╠═a42a19d6-c61e-465f-a3cc-4e9dc7eff826
# ╠═beec9c0c-d5cb-41fa-a5b6-7ba5da4afb68
# ╠═15de28c7-576f-4f80-9b57-f1112ce14cbf
# ╠═bd5408bc-9998-4bf7-9752-5823f79354f8
# ╠═0fbe4662-7777-4a23-a26c-c3f8968414b9
# ╟─17c5ddda-90ba-11eb-1fce-93b8306264fb
# ╠═0a761dc4-90bb-11eb-1f6c-fba559ed5f66
# ╠═5344b278-82b1-4a14-bf9e-350685c6d57e
# ╠═2029fc15-3ed9-4f99-b585-c93fdcdc66fb
# ╟─de41eda0-2636-4f87-b791-286a84f744ff
# ╠═0111f7f1-90c8-4ae7-ad75-68b155d4bd30
# ╠═7a86763d-fa11-48ac-94f0-7bd9874af9c5
# ╠═3553b2af-d10b-4df5-b67a-ba3d252b7e0e
# ╠═adc7cc1a-dd3d-48bc-8c34-7d30b684199d
# ╠═65718db9-5ab1-46d0-9dfb-3b234c956e1f
# ╠═28172ef1-06c6-4236-b027-b55c79ce94f1
# ╠═fdd02429-60d1-4c65-bc0d-2ea121d1a712
# ╠═119a6e70-d698-489e-9605-1757b8429f57
# ╠═1f3e4948-b75a-4d78-875f-0f8586e49e82
# ╠═c6d56e0f-0284-4c74-b632-3ae1cd1cae6a
# ╠═0a731566-69e6-409a-bb9e-9dc371dfb890
# ╠═d6f3c660-c632-4795-8bb9-35e456bebac4
# ╠═f5d2e08b-8975-4190-a033-e6630dea3ab9
# ╠═adc743b0-4174-4fa5-9c11-b30817cc3768
# ╠═b6a2b5d8-5258-4b75-8611-d25a4a075753
# ╠═e2c07c18-43c8-4899-b083-05c8180dc7d3
# ╠═78714227-094e-4a00-a72c-accfb50e1b71
# ╟─5ca39a1b-4871-4f48-9934-99c88fb504ba
# ╠═79015407-a145-44ed-853d-ca7c89676ddc
# ╟─b7ae23ec-1612-40b9-8020-21d5ee5f4c48
# ╠═b10f4fa4-7fcb-4bc1-975a-55c8079a616a
# ╠═e4b267fb-c2e9-4341-b261-860b12648127
# ╟─670121db-0989-4939-86e6-888194affb41
# ╠═92b6e763-522b-444e-9127-9ff51ef1239b
# ╠═a08ff202-3e7f-413e-8801-91d609409753
# ╠═9a5d4ff2-9ccb-4883-84fe-af3f3d52c568
# ╟─2a3caf33-8f54-42a0-ad52-a1fae4863a6a
# ╠═7edbeb65-ed32-4cab-ab20-37e65c820334
# ╠═4759682f-9239-42c9-9a30-d70dc3939285
# ╠═848c3d78-f8c6-4f44-96b2-e8f1406a56c5
# ╠═0155586a-9ffc-4319-bb93-6aeab31f670e
# ╠═4936599e-d2f5-4901-b049-6ae55cdaca24
# ╠═9054f4f9-6a31-404a-bc7e-6be1480e00c6
# ╠═37d20d35-72b9-496e-9dd0-a81a3d5c69cd
# ╟─8f446c4a-6df9-45d1-806a-3433a039229d
# ╟─f0f3347b-a89f-4ed3-94b6-1be2482fe954
# ╟─1b1bd03a-09a7-49f9-96e7-45aaa35ef8f4
# ╠═5a62989d-6b26-4209-a7c2-fde82d5a87b2
# ╟─772f74d7-097b-40e6-a8c9-ee536dcffadc
# ╠═2c8ae3f0-9e01-40b3-9cb0-0d651a048662
# ╠═5926c5c0-c552-4802-8447-58e526e00659
# ╟─f19287fa-1ed4-4994-b59a-2731f8accec0
# ╠═a6d94aa9-e4af-4dd1-a31b-b07e53a11e17
# ╠═89af27db-3cc4-424d-9741-473e863a95e4
# ╠═debff640-513f-4b1c-bddf-0cbbe33b0522
# ╠═2191de84-b685-4b25-816e-f38a6e12db3d
# ╠═85186d7a-2f4b-4766-8bcf-8edb415af4f8
# ╠═296bea20-2efa-4a1f-bc14-7d3a61e45bd2
# ╠═2baf4e94-8651-494b-8c76-b05fdba16c99
# ╠═f7e9c8c1-7939-4551-9267-28e58bb63191
# ╠═780d91c4-c606-48d5-bda6-7512f4ab2d82
# ╠═d41c401c-3ac9-47fa-8f88-c6e81f5d092d
# ╠═c2ef34f8-2ff9-4731-aed3-0262d4c0b733
# ╠═0695dd02-e617-4110-9f5e-234f5cfcad31
# ╟─395bec09-3908-4095-afdb-ade87f6b6078
# ╠═1ad7f796-e1f4-4bba-8e38-3f1ba5628474
# ╠═bc46e68f-ea1d-4f3a-b044-38bd7107bd6d
# ╠═6189ad86-5539-4ebd-9413-ad3592b24d24
# ╠═5902256e-c3e6-4194-b141-eb83e7171206
# ╠═54c2cccd-2f3d-4bc0-a7eb-4f00367ad619
# ╟─f50eeab6-7cda-4593-9f3b-6ac6dc258ba6
# ╠═5836cc48-dc6b-4768-bcb4-b7a5693e3deb
# ╠═97bff68e-54af-4b4f-8d0f-5fcda48be845
# ╠═f248ae66-6a8f-47d8-808a-8b5b3efaf795
# ╠═d93b609a-4a5b-4c5f-af43-801083d8b67f
# ╠═bb620ae9-7a38-4c39-8026-fb3a9ffb768e
# ╠═beade4cf-6b98-48d5-ad02-8102360fa55a
# ╠═024ea208-3ae1-43e8-8758-d049d0eff5d3
# ╠═1faa33a5-b679-442a-b983-16cab2b48ad9
# ╠═0fc0995f-f87a-4c13-b33a-ab01f86f3d1c
# ╠═3639bf58-809f-4221-8c91-8abbe218d0b4
# ╟─5bd4471f-d3c3-451d-94ee-05a30132e2b9
# ╠═077026fc-808f-46ef-8308-d32f50b0e12f
# ╠═1ebf3383-03c8-434d-8f09-9621344c686b
# ╠═2bd2d7da-5798-4104-89cf-a81f4d553be5
# ╠═0d27ad7a-d3a2-4873-bf5f-f6585fc63882
# ╠═1c1dd8a7-697e-45e2-9734-f221b2e46978
# ╠═53eeb659-3083-45ea-a001-23e16c29388a
# ╠═48d0afee-667b-4b9f-8610-2abb5973ef86
# ╠═8b190d6d-ed7e-4e19-9228-6ec3a8d6934e
# ╠═52a9a59f-1e3a-484f-87bd-22dbdc0d704b
# ╠═d25834fb-b564-4a54-aacb-0bac51b45e70
# ╠═aab34582-8cbb-4ff2-ab1b-d660755acbec
# ╠═30772ff5-c234-4e4b-8906-e89719be6435
# ╟─888a7310-82fe-40af-b4fd-92577fa46e4d
# ╠═63b0f25f-076c-4b0f-b09b-5ecd9d317e13
# ╠═ebfce7a9-6f8e-4977-91cc-50fb4f27c395
# ╠═619cbde6-05ec-4f2a-af3d-4bc575d0be49
# ╠═e4e852e5-1db7-4a5f-943d-ca24df4af351
# ╠═538d688e-3ee0-43a3-bfb1-bd469c065d8d
# ╠═e708275d-1b98-4127-a8a0-b6e8b4adf53b
# ╠═072c7766-f386-417d-94f6-cb135e7c1dc1
# ╠═b79ea401-e26d-4d69-96b5-544a83f8dba8
# ╠═b2d83b4d-0075-461d-ae8e-5d38badc48fe
# ╠═9652c66e-b144-4dba-9dfe-65c9d57ff339
# ╠═71007aa0-5ab4-4e69-831e-55f661572283
# ╠═1bfe2a17-5f4a-45a0-bae5-32fd61f1ccf1
# ╠═c8f3397e-fdf2-4334-98c9-96ea390aa099
# ╠═405ce3bc-124d-4815-8732-c967a01f1d66
