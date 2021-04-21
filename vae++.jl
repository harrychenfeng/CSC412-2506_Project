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

# ╔═╡ a47cc636-cbd1-4ec6-9084-9e44a436f3a8
function create_enc_dec(Dz, unpack_method)
	encoder = Chain(Dense(Ddata, Dh, tanh), Dense(Dh, Dz*2), unpack_method)
	decoder = Chain(Dense(Dz, Dh, tanh), Dense(Dh, Ddata))
	return encoder, decoder
end

# ╔═╡ 0111f7f1-90c8-4ae7-ad75-68b155d4bd30
Dz_3d = 3

# ╔═╡ adc7cc1a-dd3d-48bc-8c34-7d30b684199d
function unpack_guassian_params_3d(output)
	μ, logσ = output[1:3,:], output[4:6,:]
	return μ, logσ
end

# ╔═╡ 7a86763d-fa11-48ac-94f0-7bd9874af9c5
encoder_3d, decoder_3d = create_enc_dec(Dz_3d, unpack_guassian_params_3d)

# ╔═╡ 3553b2af-d10b-4df5-b67a-ba3d252b7e0e
function log_likelihood_larger(x,z)
  """ Compute log likelihood log_p(x|z)"""
	return sum(bernoulli_log_density(x, decoder_3d(z)),dims=1)
end

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

# ╔═╡ 8e9b5bd6-5d29-4127-8256-c71f65f50536
function visualize_samples(decoder, dim)
	plots1 = Any[]
	plots = Any[]
	for i in 1:5
		# 1. Sample five 2D/3D zs from the prior p(z)
		z = randn(dim,)
		# 2. decode each z to get logit-means
		logit_means = decoder(z)
		# 3. Transfer logit-means to Bernoulli means μ
		bern_mean = calculate_bernoulli_mean(logit_means)[1:784]
		push!(plots, draw_image(bern_mean))
		# 5. Sample 1 example from Bernoulli 
		samples1 = rand(Float64, size(bern_mean)) .< bern_mean
		push!(plots1, draw_image(samples1))
	end
	return plots1, plots
end

# ╔═╡ 012cda0a-b79e-44cc-a3c4-51c7d954de0e
function plot_mnist_image(plots, plots1)
	# 6. Display all plots in a single 10 x 4 grid
	p = plot(layout = (5,1), size=(500,800))
	for i in 1:5
		heatmap!(cat(plots[i], plots1[i], dims=2), subplot=i)
	end
	plot(p)
end

# ╔═╡ b7ae23ec-1612-40b9-8020-21d5ee5f4c48
md"""
Baseline (2D latent space)
"""

# ╔═╡ af6b3831-1de5-47e4-afaf-c9cd4adb640f
plots1_2D, plots_2D = visualize_samples(decoder, 2)

# ╔═╡ 670121db-0989-4939-86e6-888194affb41
md"""
Model with larger dimension (3D latent space)
"""

# ╔═╡ a9623ea4-e2cc-424f-8725-4c67d801ce19
plots1_3D, plots_3D = visualize_samples(decoder_3d, 3)

# ╔═╡ 0a61fd36-b757-4852-818a-a5e65d3f312f
plot_mnist_image(plots_2D, plots_3D)

# ╔═╡ 62d47455-4d48-480a-ad34-0b9cb37be6cb
plot_mnist_image(plots1_2D, plots1_3D)

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
begin
	encoder_cond = Chain(Dense(Ddata+10, Dh, tanh), Dense(Dh, Dz_3d*2), unpack_guassian_params_3d)
	decoder_cond = Chain(Dense(Dz_3d, Dh, tanh), Dense(Dh, Ddata+10))
end

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
scatter(q_μ_cond[1,:], q_μ_cond[2,:], q_μ_cond[3,:], group=labels, title="A batch 3D latent space of mean vectors for μ | labels", xlabel="z1 for mean μ", ylabel="z2 for mean μ", zlabel="z3 for mean μ")

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

# ╔═╡ 3b9c1ed3-c083-484e-89b7-253923da794e
plots1_cond, plots_cond = visualize_samples(decoder_cond, 3)

# ╔═╡ a34bc0c1-ca8c-4535-86bb-64c1ea10d1e9
plot_mnist_image(plots_cond, plots1_cond)

# ╔═╡ f50eeab6-7cda-4593-9f3b-6ac6dc258ba6
md"""
###### Semi-supervised learning
"""

# ╔═╡ 5836cc48-dc6b-4768-bcb4-b7a5693e3deb
begin
	onehot_semi = Matrix(onehot_labels)
	index = rand(1:60000,30000)
	onehot_semi[:,index] = zeros(10, 30000)
end

# ╔═╡ 97bff68e-54af-4b4f-8d0f-5fcda48be845
batches_semi = Flux.Data.DataLoader(cat(binarized_MNIST, onehot_semi, dims=1), batchsize=BS)

# ╔═╡ f248ae66-6a8f-47d8-808a-8b5b3efaf795
begin
	encoder_semi = Chain(Dense(Ddata+10, Dh, tanh), Dense(Dh, Dz_3d*2), unpack_guassian_params_3d)
	decoder_semi = Chain(Dense(Dz_3d, Dh, tanh), Dense(Dh, Ddata+10))
end

# ╔═╡ cc7274c4-1591-4b3d-a727-173243240adf
function log_likelihood_semi(x,z)
  """ Compute log likelihood log_p(x|z)"""
	# use numerically stable bernoulli
	return sum(bernoulli_log_density(x, decoder_semi(z)),dims=1)
end

# ╔═╡ dea35b3d-a22f-4765-a750-26baea97b630
joint_log_density_semi(x,z) = log_prior(z) .+ log_likelihood_semi(x,z)

# ╔═╡ bbe80124-1a79-443f-baf0-4a97ffab3693
function elbo_3d_semi(x)	
	q_μ, q_logσ = encoder_semi(x)
  	z = sample_from_var_dist_3d(q_μ, q_logσ)
  	joint_ll = joint_log_density_semi(x,z)
  	log_q_z = log_q(z, q_μ, q_logσ)
  	elbo_estimate = sum(joint_ll - log_q_z)/size(x)[2]
  	return elbo_estimate
end

# ╔═╡ dc842834-1376-40c2-86f5-174482d655e3
function loss_3d_semi(x)
  	return -elbo_3d_semi(x)
end

# ╔═╡ bb620ae9-7a38-4c39-8026-fb3a9ffb768e
function train_semi!(enc, dec, data; nepochs=100)
	params = Flux.params(enc, dec)
	opt = ADAM()
	@info "Begin training in 3D latent space with given semi labels"
	for epoch in 1:nepochs
		b_loss = 0
		for batch in data
			grads = Flux.gradient(params) do
				b_loss = loss_3d_semi(batch)
				return b_loss
			end
			Flux.Optimise.update!(opt, params, grads)
		end
		@info "Epoch $epoch: loss:$b_loss"
	end
	@info "Training in 3D latent space(semi labels) is done"
end

# ╔═╡ beade4cf-6b98-48d5-ad02-8102360fa55a
train_semi!(encoder_semi, decoder_semi, batches_semi, nepochs=3)

# ╔═╡ 6b7ce335-dca4-4e52-9f25-27d507bbf72b
plots1_semi, plots_semi = visualize_samples(decoder_semi, 3)

# ╔═╡ 5d29b1bd-8189-4054-89b1-071cac4b4fd8
plot_mnist_image(plots_semi, plots1_semi)

# ╔═╡ 5bd4471f-d3c3-451d-94ee-05a30132e2b9
md"""
### Optimizing Different Divergences
"""

# ╔═╡ 1ebf3383-03c8-434d-8f09-9621344c686b
encoder_js, decoder_js = create_enc_dec(2, unpack_guassian_params)

# ╔═╡ a25b0d00-af5a-4d80-8384-c77b57c26a0d
function log_likelihood_js(x,z)
  """ Compute log likelihood log_p(x|z)"""
	return sum(bernoulli_log_density(x, decoder_js(z)),dims=1)
end

# ╔═╡ 390781e2-575a-4c33-9e9d-439c6d06d60d
joint_log_density_js(x,z) = log_prior(z) .+ log_likelihood_js(x,z)

# ╔═╡ 2bd2d7da-5798-4104-89cf-a81f4d553be5
function elbo_js(x)	
	q_μ, q_logσ = encoder_js(x)
  	z = sample_from_var_dist(q_μ, q_logσ)
  	joint_ll = joint_log_density_js(x,z)
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
train_js!(encoder_js, decoder_js, batches, nepochs=3)

# ╔═╡ d25834fb-b564-4a54-aacb-0bac51b45e70
plots1_js, plots_js = visualize_samples(decoder_js, 2)

# ╔═╡ 30772ff5-c234-4e4b-8906-e89719be6435
plot_mnist_image(plots_js, plots1_js)

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
encoder_beta, decoder_beta = create_enc_dec(2, unpack_guassian_params)

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
  	elbo_estimate = mean(joint_ll - log_q_z)
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
train_beta!(encoder_beta, decoder_beta, float_batches, nepochs=5)

# ╔═╡ c8f3397e-fdf2-4334-98c9-96ea390aa099
plots1_beta, plots_beta = visualize_samples(decoder_beta, 2)

# ╔═╡ 405ce3bc-124d-4815-8732-c967a01f1d66
plot_mnist_image(plots_beta, plots1_beta)

# ╔═╡ 90677b4f-d4eb-46c6-af37-9abd26b3a925
md"""
### Inference
Use the baseline model to infer the bottom of a digit given the top
"""

# ╔═╡ 6d6b50a5-5ee5-4694-8470-f4cc6838c0e9
Dhalf = Int(28*28/2)

# ╔═╡ 6efb46a3-2d03-4f61-b85d-f2f34d0a04f3
# Helper function for drawing only top half of the MNIST digit in 28*28 shape
function draw_top_half_image(x)
	x = reshape(x, 28, 28, :)
	bot_x = x[1:14,:,:]
	return reshape(bot_x, (14,28))
end

# ╔═╡ 8357164e-e220-4911-bdbb-442586fb83ec
# Helper function for drawing only top half of the MNIST digit in 28*28 shape
function draw_bot_half_image(x)
	x = reshape(x, 28, 28, :)
	bot_x = x[15:28,:,:]
	return reshape(bot_x, (14,28))
end

# ╔═╡ 5a4c50e1-c849-4d54-95d8-3d5c3e27f439
# Calculate log likelihood log_p(top|z)
function log_p_top_z(top, z)
	x̂_half = decoder(z)[1:Dhalf,:]
	return sum(bernoulli_log_density(top,x̂_half),dims=1)
end

# ╔═╡ 60b5c2ea-5d6d-4e64-839c-ce5e7a83cf27
md"""
Log joint density $p(z,top)$
"""

# ╔═╡ 3f8b673f-b9fb-435e-9832-321fd2539846
# Calculate log_p(top, z)
joint_log_density_top(top,z) = log_prior(z) .+ log_p_top_z(top,z)

# ╔═╡ 393f1d41-99d5-4f25-8b9d-bfa8b57ec65d
md"""
Stochastic variational inference $p(z|top)$
"""

# ╔═╡ 925453b6-8d02-491a-865e-cd464784ca15
q_μ_top = randn(Dz,1)

# ╔═╡ fd71a120-e07f-4430-bc9b-3001eea4d0c8
q_logσ_top = randn(Dz,1)

# ╔═╡ a01ebb83-355d-46a7-b6ea-5cbea7485b62
function elbo_top(top, q_μ_top, q_logσ_top)
  	z = sample_from_var_dist(q_μ_top, q_logσ_top)
  	joint_ll = joint_log_density_top(top,z)
  	log_q_z = log_q(z, q_μ_top, q_logσ_top)
  	elbo_estimate = mean(joint_ll - log_q_z)
  	return -elbo_estimate
end

# ╔═╡ 815aded7-fc7d-4c41-91a9-3251ec5cf3f4
function loss_top(top, q_μ_top, q_logσ_top)
  	return -elbo_top(top, q_μ_top, q_logσ_top)
end

# ╔═╡ f5bc9db5-deb8-476f-9439-ef03b3707dd3
n = size(train_labels)[1]

# ╔═╡ b6e0fdd6-65bd-4259-906e-e57a6e928437
# Construct a dataset consists of digit 0
indices_0 = [i for i in 1:n if train_labels[i]==0]

# ╔═╡ 85202977-9f9d-4c4e-b36f-6e86fa12c66e
digit_0 = binarized_MNIST[:,indices_0]

# ╔═╡ d500888b-2718-4df3-9ba2-5886f70f993b
function train_top!(q_μ, q_logσ, data, loss_func; nepochs=100)
	params = Flux.params(q_μ, q_logσ)
	opt = ADAM()
	@info "Begin training to optimize q_μ and q_logσ"
	for epoch in 1:nepochs
		b_loss = 0
		grads = Flux.gradient(params) do
			b_loss = loss_func(data[1:Dhalf])
			return b_loss
		end
		Flux.Optimise.update!(opt, params, grads)
		@info "Epoch $epoch: loss:$b_loss"
	end
	@info "Optimizing q_μ and q_logσ is done"
end

# ╔═╡ 4a2639c0-5b41-4d8f-8dc3-9eee168939a0
loss_tophalf(top) = loss_top(top, q_μ_top, q_logσ_top)	

# ╔═╡ 1659084a-15e8-4681-98d3-e5febba12364
size(digit_0)

# ╔═╡ 27b64959-d018-414d-9f43-5b729a1172d0
md"""
Take digit $0$ and infer the bottom part given the top part
"""

# ╔═╡ feb10319-00b1-4acb-acd1-6abf78ecd6e3
test_img = train_digits[2]

# ╔═╡ 3768d31d-dadc-4be8-afc3-2b16ea4cc167
train_top!(q_μ_top, q_logσ_top, digit_0[:,2], loss_tophalf, nepochs=2)

# ╔═╡ 53a5b0df-9cc8-4164-84b0-a73498708c69
begin
	p_top = plot(layout = (1,2))
	
	# Take a sample z from the approximate posterior
	z_top = sample_from_var_dist(q_μ_top, q_logσ_top)
	# Feed z to decoder
	logits_mean_top = decoder(z_top)
	# Convert to bernoulli mean
	bern_mean_top = calculate_bernoulli_mean(logits_mean_top)
	bot_part = draw_bot_half_image(bern_mean_top)
	top_part = draw_top_half_image(test_img)
	cat_img = cat(top_part, bot_part, dims=1)
	
	# Plot original and inferred results
	plot!(test_img, title="Original digit 0", subplot=1)
	plot!(draw_image(vec(cat_img)), title= "Inferred digit 0", subplot=2)
	
	plot(p_top)
end

# ╔═╡ 986442a3-c767-4067-b9a5-30c010d5b4fc
md"""
### More interesting data
Train the VAE model on Fashion MNIST dataset
"""

# ╔═╡ 172202fa-bd2a-47c5-a278-108d21cb742a
train_fashion = Flux.Data.FashionMNIST.images(:train)

# ╔═╡ a4fb2a32-87e2-4068-a10f-85f38405000b
greyscale_fashion = hcat(float.(reshape.(train_fashion,:))...)

# ╔═╡ 435ac5c8-f763-4d87-a06d-d28183f9c2ab
binarized_fashion = greyscale_fashion .> 0.5

# ╔═╡ 410be0e5-99ea-4cbf-a24c-1a9d8118ff6d
fashion_batches = Flux.Data.DataLoader(binarized_fashion, batchsize=BS)

# ╔═╡ 5616d1c2-9fdd-4d17-94bb-a3afa38e3cc0
encoder_fashion, decoder_fashion = create_enc_dec(Dz_3d, unpack_guassian_params_3d)

# ╔═╡ 71b32858-4c7d-4864-a839-eee9cc11eabe
function log_likelihood_fashion(x,z)
	return sum(bernoulli_log_density(x, decoder_fashion(z)),dims=1)
end

# ╔═╡ 222c76dd-b44c-448c-99fc-402ccdb739ad
joint_log_density_fashion(x,z) = log_prior(z) .+ log_likelihood_fashion(x,z)

# ╔═╡ 266c0e2e-f46c-43ef-8e6c-641eaf57c5b2
function elbo_fashion(x)
	q_μ, q_logσ = encoder_fashion(x)
  	z = sample_from_var_dist_3d(q_μ, q_logσ)
  	joint_ll = joint_log_density_fashion(x,z)
  	log_q_z = log_q(z, q_μ, q_logσ)
  	elbo_estimate = mean(joint_ll - log_q_z)
	return elbo_estimate
end

# ╔═╡ 1595c464-7d09-436a-809e-bb2c13faecb8
function loss_fashion(x)
  	return -elbo_fashion(x)
end

# ╔═╡ 88fcf777-dd27-4333-bf2c-94ed06eac788
function train_fashion!(enc, dec, data; nepochs=100)
	params = Flux.params(enc, dec)
	opt = ADAM()
	@info "Begin training on FashionMNIST in 3D latent space"
	for epoch in 1:nepochs
		b_loss = 0
		for batch in data
			grads = Flux.gradient(params) do
				b_loss = loss_fashion(batch)
				return b_loss
			end
			Flux.Optimise.update!(opt, params, grads)
		end
		@info "Epoch $epoch: loss:$b_loss"
	end
	@info "Training on FashionMNIST in 3D latent space is done"
end

# ╔═╡ 7ec721d8-a1ff-4598-804f-c5380df78361
train_fashion!(encoder_fashion, decoder_fashion, fashion_batches, nepochs=5)

# ╔═╡ 3ea00da5-e845-4f0f-912f-5a4d7d6f1ebf
plots1_fashion, plots_fashion = visualize_samples(decoder_fashion, 3)

# ╔═╡ fd1e8ee6-9edb-4990-b496-37289dd6785d
plot_mnist_image(plots_fashion, plots1_fashion)

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
# ╠═a47cc636-cbd1-4ec6-9084-9e44a436f3a8
# ╠═0111f7f1-90c8-4ae7-ad75-68b155d4bd30
# ╠═adc7cc1a-dd3d-48bc-8c34-7d30b684199d
# ╠═7a86763d-fa11-48ac-94f0-7bd9874af9c5
# ╠═3553b2af-d10b-4df5-b67a-ba3d252b7e0e
# ╠═28172ef1-06c6-4236-b027-b55c79ce94f1
# ╠═fdd02429-60d1-4c65-bc0d-2ea121d1a712
# ╠═119a6e70-d698-489e-9605-1757b8429f57
# ╠═1f3e4948-b75a-4d78-875f-0f8586e49e82
# ╠═c6d56e0f-0284-4c74-b632-3ae1cd1cae6a
# ╠═0a731566-69e6-409a-bb9e-9dc371dfb890
# ╠═d6f3c660-c632-4795-8bb9-35e456bebac4
# ╠═b6a2b5d8-5258-4b75-8611-d25a4a075753
# ╠═e2c07c18-43c8-4899-b083-05c8180dc7d3
# ╠═78714227-094e-4a00-a72c-accfb50e1b71
# ╟─5ca39a1b-4871-4f48-9934-99c88fb504ba
# ╠═79015407-a145-44ed-853d-ca7c89676ddc
# ╠═8e9b5bd6-5d29-4127-8256-c71f65f50536
# ╠═012cda0a-b79e-44cc-a3c4-51c7d954de0e
# ╟─b7ae23ec-1612-40b9-8020-21d5ee5f4c48
# ╠═af6b3831-1de5-47e4-afaf-c9cd4adb640f
# ╟─670121db-0989-4939-86e6-888194affb41
# ╠═a9623ea4-e2cc-424f-8725-4c67d801ce19
# ╠═0a61fd36-b757-4852-818a-a5e65d3f312f
# ╠═62d47455-4d48-480a-ad34-0b9cb37be6cb
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
# ╟─772f74d7-097b-40e6-a8c9-ee536dcffadc
# ╠═2c8ae3f0-9e01-40b3-9cb0-0d651a048662
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
# ╠═6189ad86-5539-4ebd-9413-ad3592b24d24
# ╠═3b9c1ed3-c083-484e-89b7-253923da794e
# ╠═a34bc0c1-ca8c-4535-86bb-64c1ea10d1e9
# ╟─f50eeab6-7cda-4593-9f3b-6ac6dc258ba6
# ╠═5836cc48-dc6b-4768-bcb4-b7a5693e3deb
# ╠═97bff68e-54af-4b4f-8d0f-5fcda48be845
# ╠═f248ae66-6a8f-47d8-808a-8b5b3efaf795
# ╠═cc7274c4-1591-4b3d-a727-173243240adf
# ╠═dea35b3d-a22f-4765-a750-26baea97b630
# ╠═bbe80124-1a79-443f-baf0-4a97ffab3693
# ╠═dc842834-1376-40c2-86f5-174482d655e3
# ╠═bb620ae9-7a38-4c39-8026-fb3a9ffb768e
# ╠═beade4cf-6b98-48d5-ad02-8102360fa55a
# ╠═6b7ce335-dca4-4e52-9f25-27d507bbf72b
# ╠═5d29b1bd-8189-4054-89b1-071cac4b4fd8
# ╟─5bd4471f-d3c3-451d-94ee-05a30132e2b9
# ╠═1ebf3383-03c8-434d-8f09-9621344c686b
# ╠═a25b0d00-af5a-4d80-8384-c77b57c26a0d
# ╠═390781e2-575a-4c33-9e9d-439c6d06d60d
# ╠═2bd2d7da-5798-4104-89cf-a81f4d553be5
# ╠═0d27ad7a-d3a2-4873-bf5f-f6585fc63882
# ╠═1c1dd8a7-697e-45e2-9734-f221b2e46978
# ╠═53eeb659-3083-45ea-a001-23e16c29388a
# ╠═d25834fb-b564-4a54-aacb-0bac51b45e70
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
# ╟─90677b4f-d4eb-46c6-af37-9abd26b3a925
# ╠═6d6b50a5-5ee5-4694-8470-f4cc6838c0e9
# ╠═6efb46a3-2d03-4f61-b85d-f2f34d0a04f3
# ╠═8357164e-e220-4911-bdbb-442586fb83ec
# ╠═5a4c50e1-c849-4d54-95d8-3d5c3e27f439
# ╟─60b5c2ea-5d6d-4e64-839c-ce5e7a83cf27
# ╠═3f8b673f-b9fb-435e-9832-321fd2539846
# ╟─393f1d41-99d5-4f25-8b9d-bfa8b57ec65d
# ╠═925453b6-8d02-491a-865e-cd464784ca15
# ╠═fd71a120-e07f-4430-bc9b-3001eea4d0c8
# ╠═a01ebb83-355d-46a7-b6ea-5cbea7485b62
# ╠═815aded7-fc7d-4c41-91a9-3251ec5cf3f4
# ╠═f5bc9db5-deb8-476f-9439-ef03b3707dd3
# ╠═b6e0fdd6-65bd-4259-906e-e57a6e928437
# ╠═85202977-9f9d-4c4e-b36f-6e86fa12c66e
# ╠═d500888b-2718-4df3-9ba2-5886f70f993b
# ╠═4a2639c0-5b41-4d8f-8dc3-9eee168939a0
# ╠═1659084a-15e8-4681-98d3-e5febba12364
# ╟─27b64959-d018-414d-9f43-5b729a1172d0
# ╠═feb10319-00b1-4acb-acd1-6abf78ecd6e3
# ╠═3768d31d-dadc-4be8-afc3-2b16ea4cc167
# ╠═53a5b0df-9cc8-4164-84b0-a73498708c69
# ╟─986442a3-c767-4067-b9a5-30c010d5b4fc
# ╠═172202fa-bd2a-47c5-a278-108d21cb742a
# ╠═a4fb2a32-87e2-4068-a10f-85f38405000b
# ╠═435ac5c8-f763-4d87-a06d-d28183f9c2ab
# ╠═410be0e5-99ea-4cbf-a24c-1a9d8118ff6d
# ╠═5616d1c2-9fdd-4d17-94bb-a3afa38e3cc0
# ╠═71b32858-4c7d-4864-a839-eee9cc11eabe
# ╠═222c76dd-b44c-448c-99fc-402ccdb739ad
# ╠═266c0e2e-f46c-43ef-8e6c-641eaf57c5b2
# ╠═1595c464-7d09-436a-809e-bb2c13faecb8
# ╠═88fcf777-dd27-4333-bf2c-94ed06eac788
# ╠═7ec721d8-a1ff-4598-804f-c5380df78361
# ╠═3ea00da5-e845-4f0f-912f-5a4d7d6f1ebf
# ╠═fd1e8ee6-9edb-4990-b496-37289dd6785d
