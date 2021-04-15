### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ d402633e-8c18-11eb-119d-017ad87927b0
using Flux

# ╔═╡ c70eaa72-90ad-11eb-3600-016807d53697
using StatsFuns: log1pexp #log(1 + exp(x))

# ╔═╡ 4a8432e4-5a71-4e2d-96be-3ee4061c42f6
using BSON

# ╔═╡ 0a761dc4-90bb-11eb-1f6c-fba559ed5f66
using Plots ##

# ╔═╡ 587b1628-2db2-404e-987c-a7bdaa0bcf51
using Plots: scatter

# ╔═╡ 5344b278-82b1-4a14-bf9e-350685c6d57e
using Images

# ╔═╡ 0155586a-9ffc-4319-bb93-6aeab31f670e
using Statistics

# ╔═╡ 5a62989d-6b26-4209-a7c2-fde82d5a87b2
using ConditionalDists

# ╔═╡ 9e368304-8c16-11eb-0417-c3792a4cd8ce
md"""
# Assignment 3: Variational Autoencoders

- Student Name: Feng Chen
- Student #: $1002252956$
- Collaborators: Just myself

## Background

In this assignment we will implement and investigate a Variational Autoencoder as introduced by Kingma and Welling in [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).


### Data: Binarized MNIST

In this assignment we will consider an  MNIST dataset of $28\times 28$ pixel images where each pixel is **either on or off**.

The binary variable $x_i \in \{0,1\}$ indicates whether the $i$-th pixel is off or on.

Additionally, we also have a digit label $y \in \{0, \dots, 9\}$. Note that we will not use these labels for our generative model. We will, however, use them for our investigation to assist with visualization.

### Tools

In previous assignments you were required to implement a simple neural network and gradient descent manually. In this assignment you are permitted to use a machine learning library for convenience functions such as optimizers, neural network layers, initialization, dataloaders.

However, you **may not use any probabilistic modelling elements** implemented in these frameworks. You cannot use `Distributions.jl` or any similar software. In particular, sampling from and evaluating probability densities under distributions must be written explicitly by code written by you or provided in starter code.
"""

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

# ╔═╡ 2093080c-8e85-11eb-1cdb-b35eb40e3949
md"""
## Model Definition

Each element in the data $x \in D$ is a vector of $784$ pixels. 
Each pixel $x_d$ is either on, $x_d = 1$ or off $x_d = 0$.

Each element corresponds to a handwritten digit $\{0, \dots, 9\}$.
Note that we do not observe these labels, we are *not* training a supervised classifier.

We will introduce a latent variable $z \in \mathbb{R}^2$ to represent the digit.
The dimensionality of this latent space is chosen so we can easily visualize the learned features. A larger dimensionality would allow a more powerful model. 


- **Prior**: The prior over a digit's latent representation is a multivariate standard normal distribution. $p(z) = \mathcal{N}(z \mid \mathbf{0}, \mathbf{1})$
- **Likelihood**: Given a latent representation $z$ we model the distribution over all 784 pixels as the product of independent Bernoulli distributions parametrized by the output of the "decoder" neural network $f_\theta(z)$.
```math
p_\theta(x \mid z) = \prod_{d=1}^{784} \text{Ber}(x_d \mid f_\theta(z)_d)
```

### Model Parameters

Learning the model will involve optimizing the parameters $\theta$ of the "decoder" neural network, $f_\theta$. 

You may also use library provided layers such as `Dense` [as described in the documentation](https://fluxml.ai/Flux.jl/stable/models/basics/#Stacking-It-Up-1). 

Note that, like many neural network libraries, Flux avoids explicitly providing parameters as function arguments, i.e. `neural_net(z)` instead of `neural_net(z, params)`.

You can access the model parameters `params(neural_net)` for taking gradients `gradient(()->loss(data), params(neural_net))` and updating the parameters with an [Optimiser](https://fluxml.ai/Flux.jl/stable/training/optimisers/).

However, if all this is too fancy feel free to continue using your implementations of simple neural networks and gradient descent from previous assignments.

"""

# ╔═╡ 45bc7e00-90ac-11eb-2d62-092a13dd1360
md"""
### Numerical Stability

The Bernoulli distribution $\text{Ber}(x \mid \mu)$ where $\mu \in [0,1]$ is difficult to optimize for a few reasons.

We prefer unconstrained parameters for gradient optimization. This suggests we might want to transform our parameters into an unconstrained domain, e.g. by parameterizing the `log` parameter.

We also should consider the behaviour of the gradients with respect to our parameter, even under the transformation to unconstrained domain. For instance a poor transformation might encourage optimization into regions where gradient magnitude vanishes. This is often called "saturation".

For this reasons we should use a numerically stable transformation of the Bernoulli parameters. 
One solution is to parameterize the "logit-means": $y = \log(\frac{\mu}{1-\mu})$.

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
		# Optional: save trained parameters to avoid retraining later
	end
	# return nothing, this mutates the parameters of enc and dec!
end

# ╔═╡ c86a877c-90b9-11eb-31d8-bbcb71e4fa66
# train!(encoder, decoder, batches, nepochs=10)

# ╔═╡ 74171598-5b10-4449-aabb-49b1e168646b
# using BSON: @save

# ╔═╡ beec9c0c-d5cb-41fa-a5b6-7ba5da4afb68
# @save "encoder.bson" encoder 

# ╔═╡ 1c407b72-334e-45a9-8095-da6abff17b20
# @save "decoder.bson" decoder 

# ╔═╡ bd5408bc-9998-4bf7-9752-5823f79354f8
BSON.load("encoder.bson", @__MODULE__)

# ╔═╡ 7a17bdb1-28f3-4bc3-b28f-b5b71de39088
BSON.load("decoder.bson", @__MODULE__)

# ╔═╡ 17c5ddda-90ba-11eb-1fce-93b8306264fb
md"""
## Visualizing the Model Learned Representation

We will use the model and variational networks to visualize the latent representations of our data learned by the model.

We will use a variatety of qualitative techniques to get a sense for our model by generating distributions over our data, sampling from them, and interpolating in the latent space.
"""

# ╔═╡ 1201bfee-90bb-11eb-23e5-af9a61f64679
md"""
### 1. Latent Distribution of Batch

1. Use `encoder` to produce a batch of latent parameters $\mu, \log \sigma$
2. Take the 2D mean vector $\mu$ for each latent parameter in the batch.
3. Plot these mean vectors in the 2D latent space with a scatterplot
4. Colour each point according to its "digit class label" 0 to 9.
5. Display a single colourful scatterplot
"""

# ╔═╡ d908c2f4-90bb-11eb-11b1-b340f58a1584
# 1. Use encoder to produce a batch of latent parameters μ, logσ
# Take the first batch
q_μ, q_logσ = encoder(first(batches))

# ╔═╡ 9676431f-4668-4f66-a7a3-287910bf8c7e
begin
	train_labels = Flux.Data.MNIST.labels(:train)
	label_batches = Flux.Data.DataLoader(train_labels, batchsize=BS)
	labels = first(label_batches)
end

# ╔═╡ db18e7e2-90bb-11eb-18e5-87e4f094123d
scatter(q_μ[1,:], q_μ[2,:], group=labels, title="A batch latent space of mean vectors for μ", xlabel="z1 for mean μ", ylabel="z2 for mean μ")

# ╔═╡ d9c5399b-11cc-43db-85ea-f2b5d1f447d8
savefig("Q1_latent_space.png")

# ╔═╡ dcedbba4-90bb-11eb-2652-bf6448095107
md"""
### 2. Visualizing Generative Model of Data

1. Sample 10 $z$ from the prior $p(z)$.
2. Use the model to decode each $z$ to the distribution logit-means over $x$.
3. Transform the logit-means to the Bernoulli means $\mu$. (this does not need to be efficient)
4. For each $z$, visualize the $\mu$ as a $28 \times 28$ greyscale images.
5. For each $z$, sample 3 examples from the Bernoulli likelihood $x \sim \text{Bern}(x \mid \mu(z))$.
6. Display all plots in a single 10 x 4 grid. Each row corresponding to a sample $z$. Do not include any axis labels.
"""

# ╔═╡ 8e6b3b1e-bdb5-4458-b6fc-98dad7349cb0
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

# ╔═╡ 805a265e-90be-11eb-2c34-1dd0cd1a968c
begin
	# 1. Sample 10 z from the prior p(z)
	zs = Any[]
	for i in 1:10
		sample_z = randn(2,)
		push!(zs, sample_z)
	end
end

# ╔═╡ 80d6b61a-90be-11eb-2fae-638cdaaf7abd
begin
	# 2. decode each z to get logit-means
	plots1, plots2, plots3 = Any[], Any[], Any[]
	plots = Any[]
	for i in 1:10
		logit_means = decoder(zs[i])
		# 3. Transfer logit-means to Bernoulli means μ
		bern_mean = exp.(logit_means) ./ (1 .+ exp.(logit_means))
		
		push!(plots, draw_image(bern_mean))
		
		# 5. Sample 3 examples from Bernoulli 
		samples1 = rand(Float64, size(bern_mean)) .< bern_mean
		push!(plots1, draw_image(samples1))
		
		samples2 =  rand(Float64, size(bern_mean)) .< bern_mean
		push!(plots2, draw_image(samples2))
		
		samples3 = rand(Float64, size(bern_mean)) .< bern_mean
		push!(plots3, draw_image(samples3))
	end
end

# ╔═╡ f1a95304-6395-4f59-96dc-c62601255a77
begin
	# 6. Display all plots in a single 10 x 4 grid
	p = plot(layout = (10,1), size=(500,1200))
	for i in 1:10
		heatmap!(cat(plots[i], plots1[i], plots2[i], plots3[i], dims=2), subplot=i)
	end
	plot(p)
end

# ╔═╡ 868abcc7-28d8-433e-ac4f-51cc8c5bb848
savefig("Q2_visualize.png")

# ╔═╡ 82b0a368-90be-11eb-0ddb-310f332a83f0
md"""
### 3. Visualizing Regenerative Model and Reconstruction

1. Sample 4 digits from the data $x \sim \mathcal{D}$
2. Encode each digit to a latent distribution $q_\phi(z)$
3. For each latent distribution, sample 2 representations $z \sim q_\phi$
4. Decode each $z$ and transform to the Bernoulli means $\mu$
5. For each $\mu$, sample 1 "reconstruction" $\hat x \sim \text{Bern}(x \mid \mu)$
6. For each digit $x$ display (28x28) greyscale images of $x, \mu, \hat x$
"""

# ╔═╡ f27bdffa-90c0-11eb-0f71-6d572f799290
begin
	# 1. Sample 4 digits from the data x~D
	sample2_i = rand(1:60000, 4)
	binarized_xs = binarized_MNIST[:,sample2_i]
	# 2. Encode each digit to a latent distribution 
	sample2_μ, sample2_logσ = encoder(binarized_xs)
end

# ╔═╡ edef7f53-02a3-4841-a9e3-56706f0e5ef4
begin
	xs = Any[]
	for i in 1:4
		push!(xs, draw_image(binarized_xs[:,i]))
	end
end

# ╔═╡ 00b7f55e-90c1-11eb-119e-f577037923a9
begin
	μs1, μs2, x̂s1, x̂s2 = Any[], Any[], Any[], Any[]
	for i in 1:4
		# 3. For each latent distribution, sample 2 representations z~q
		z1 = sample_from_var_dist(sample2_μ[:,i], sample2_logσ[:,i])
		z2 = sample_from_var_dist(sample2_μ[:,i], sample2_logσ[:,i])
		
		# 4. Decode each z and transform to the Bernoulli means μ
		logit_means1 = decoder(z1)
		bern_mean1 = exp.(logit_means1) ./ (1 .+ exp.(logit_means1))
		push!(μs1, draw_image(bern_mean1))
		
		logit_means2 = decoder(z2)
		bern_mean2 = exp.(logit_means2) ./ (1 .+ exp.(logit_means2))
		push!(μs2, draw_image(bern_mean2))
		
		# 5. For each μ, sample 1 "reconstruction" x_hat ~ Bern(x|μ)
		x̂1 = rand(Float64, size(bern_mean1)) .< bern_mean1
		push!(x̂s1, draw_image(x̂1))
		
		x̂2 = rand(Float64, size(bern_mean2)) .< bern_mean2
		push!(x̂s2, draw_image(x̂2))
	end
end

# ╔═╡ f41c2f1d-f6f9-49e0-aa20-73b2ce55bf7f
begin
	# 6. For each digit x display (28*28) greyscale images of x, μ, x̂
	p2 = plot(layout = (4,1))
	# row = 1
	for i in 1:4
		heatmap!(cat(xs[i], μs1[i], μs2[i], x̂s1[i], x̂s2[i], dims=2), subplot=i)
	end
	plot(p2)
end

# ╔═╡ cf4d6579-52a5-45bc-81ed-6797f59a2846
savefig("Q3_visualize.png")

# ╔═╡ 02181adc-90c1-11eb-29d7-736dce72a0ac
md"""
### 4. Latent Interpolation Along Lattice

1. Produce a $50 \times 50$ "lattice" or collection of cartesian coordinates $z = (z_x, z_y) \in \mathbb{R}^2$.
2. For each $z$, decode and transform to a 28x28 greyscale image of the Bernoulli means $\mu$
3. Each point in the `50x50` latent lattice corresponds now to a `28x28` greyscale image. Concatenate all these images appropriately.
4. Display a single `1400x1400` pixel greyscale image corresponding to the learned latent space.
"""

# ╔═╡ 3a0e1d5a-90c2-11eb-16a7-8f9de1ea09e4
# 1. Produce a 50 * 50 lattice
begin
	random_μ_logσ = rand(-1:2/50:1, (2,1,50,50))
	random_μ_logσ2 = rand(-1:2/50:1, (2,1,50,50))
	
	zx = sample_from_var_dist(random_μ_logσ[1,:,:,:], random_μ_logσ[2,:,:,:])
	zy = sample_from_var_dist(random_μ_logσ2[1,:,:,:], random_μ_logσ2[2,:,:,:])
	lattice = cat(zx, zy, dims=1)
end

# ╔═╡ 3a9a2624-90c2-11eb-1986-17b80a2a58c5
# 2. For each z, decode and transform to a 28 * 28 greyscale image of the Bernoulli means μ
begin
	plots_lattice = Any[]
	for i in 1:50
		for j in 1:50
			logit_means_lattice = decoder(lattice[:,i,j])
			bern_mean_lattice = exp.(logit_means_lattice) ./ (1 .+ exp.(logit_means_lattice))
			push!(plots_lattice, draw_image(bern_mean_lattice))
		end
	end
end

# ╔═╡ 3b6f8e5e-90c2-11eb-3da4-a5fd3048ab63
# 3. Each point in the 50x50 latent lattice corresponds now to a 28x28 greyscale image. Concatenate all these images appropriately.
begin
	p3 = plot(layout = (50,1),size=(700,700))
	for i in 1:50
		start_i = (i-1)*50+1
		next_i = start_i+1
		cat_res = cat(plots_lattice[start_i],plots_lattice[next_i], dims=2)
		for j in 3:50
			cat_res = cat(cat_res, plots_lattice[(i-1)*50+j], dims=2)
		end
		heatmap!(cat_res, subplot=i)
		car_res = Any[]
	end
	# 4. Display a single 1400x1400 pixel greyscale image corresponding to the learned latent space.
	plot(p3)
end

# ╔═╡ 4a43c8cc-b2f6-4364-b5f3-8bd58e3f2f2e
savefig("Q4_visualize.png")

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
  #TODO variational parameters from data	
  q_μ, q_logσ = encoder_3d(x)
  #TODO: sample from variational distribution
  z = sample_from_var_dist_3d(q_μ, q_logσ)
  #TODO: joint likelihood of z and x under model
  joint_ll = joint_log_density_3d(x,z)
  #TODO: likelihood of z under variational distribution
  log_q_z = log_q(z, q_μ, q_logσ)
  #TODO: Scalar value, mean variational evidence lower bound over batch
  elbo_estimate = sum(joint_ll - log_q_z)/size(x)[2]
  # return logσ for plotting
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
	# return nothing, this mutates the parameters of enc and dec!
	# return logσs
end

# ╔═╡ d6f3c660-c632-4795-8bb9-35e456bebac4
train_3d!(encoder_3d, decoder_3d, batches, nepochs=3)

# ╔═╡ f5d2e08b-8975-4190-a033-e6630dea3ab9
# begin
# 	@save "encoder_3d.bson" encoder_3d 
# 	@save "decoder_3d.bson" decoder_3d
# end

# ╔═╡ adc743b0-4174-4fa5-9c11-b30817cc3768
begin
	BSON.load("encoder_3d.bson", @__MODULE__)
	BSON.load("decoder_3d.bson", @__MODULE__)
end

# ╔═╡ b6a2b5d8-5258-4b75-8611-d25a4a075753
q_μ_3d, q_logσ_3d = encoder_3d(first(batches))

# ╔═╡ 78714227-094e-4a00-a72c-accfb50e1b71
scatter(q_μ_3d[1,:], q_μ_3d[2,:], q_μ_3d[3,:], group=labels, title="A batch latent space of mean vectors for μ", xlabel="z1 for mean μ", ylabel="z2 for mean μ", zlabel="z3 for mean μ")

# ╔═╡ 5ca39a1b-4871-4f48-9934-99c88fb504ba
md"""
###### Comparison with baselines
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
		bern_mean_3d = exp.(logit_means_3d) ./ (1 .+ exp.(logit_means_3d))
		
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

# ╔═╡ 5926c5c0-c552-4802-8447-58e526e00659
#decoder_cond = Chain(Dense(Dz_3d, Dh, tanh), Dense(Dh, Ddata))

# ╔═╡ 2c8ae3f0-9e01-40b3-9cb0-0d651a048662
# encoder_cond = Chain(Dense(Ddata, Dh, tanh), Dense(Dh, Dz_3d*2), unpack_guassian_params_3d)

# ╔═╡ f7e9c8c1-7939-4551-9267-28e58bb63191
# function train_conditional!(enc, dec, data; nepochs=100)
# 	params = Flux.params(enc, dec)
# 	opt = ADAM()
# 	@info "Begin training in 3D latent space"
# 	for epoch in 1:nepochs
# 		b_loss = 0
# 		logσ = Any[]
# 		for batch in data
# 			grads = Flux.gradient(params) do
# 				b_loss, logσ = loss_3d(batch)
# 				return b_loss
# 			end
# 			Flux.Optimise.update!(opt, params, grads)
# 		end
# 		@info "Epoch $epoch: loss:$b_loss"
# 	end
# 	# return nothing, this mutates the parameters of enc and dec!
# end

# ╔═╡ 780d91c4-c606-48d5-bda6-7512f4ab2d82
# train_conditional!(encoder_3d, decoder_3d, batches, nepochs=3)

# ╔═╡ Cell order:
# ╟─9e368304-8c16-11eb-0417-c3792a4cd8ce
# ╠═d402633e-8c18-11eb-119d-017ad87927b0
# ╠═54749c92-8c1d-11eb-2a54-a1ae0b1dc587
# ╠═176f0938-8c1e-11eb-1135-a5db6781404d
# ╠═c6fa2a9c-8c1e-11eb-3e3c-9f8f5c218dec
# ╠═9e7e46b0-8e84-11eb-1648-0f033e4e6068
# ╠═743d473c-8c1f-11eb-396d-c92cacb0235b
# ╠═db655546-8e84-11eb-21df-25f7c8e82362
# ╟─2093080c-8e85-11eb-1cdb-b35eb40e3949
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
# ╠═74171598-5b10-4449-aabb-49b1e168646b
# ╠═beec9c0c-d5cb-41fa-a5b6-7ba5da4afb68
# ╠═1c407b72-334e-45a9-8095-da6abff17b20
# ╠═4a8432e4-5a71-4e2d-96be-3ee4061c42f6
# ╠═bd5408bc-9998-4bf7-9752-5823f79354f8
# ╠═7a17bdb1-28f3-4bc3-b28f-b5b71de39088
# ╟─17c5ddda-90ba-11eb-1fce-93b8306264fb
# ╠═0a761dc4-90bb-11eb-1f6c-fba559ed5f66
# ╟─1201bfee-90bb-11eb-23e5-af9a61f64679
# ╠═d908c2f4-90bb-11eb-11b1-b340f58a1584
# ╠═587b1628-2db2-404e-987c-a7bdaa0bcf51
# ╠═9676431f-4668-4f66-a7a3-287910bf8c7e
# ╠═db18e7e2-90bb-11eb-18e5-87e4f094123d
# ╠═d9c5399b-11cc-43db-85ea-f2b5d1f447d8
# ╟─dcedbba4-90bb-11eb-2652-bf6448095107
# ╠═5344b278-82b1-4a14-bf9e-350685c6d57e
# ╠═8e6b3b1e-bdb5-4458-b6fc-98dad7349cb0
# ╠═805a265e-90be-11eb-2c34-1dd0cd1a968c
# ╠═80d6b61a-90be-11eb-2fae-638cdaaf7abd
# ╠═f1a95304-6395-4f59-96dc-c62601255a77
# ╠═868abcc7-28d8-433e-ac4f-51cc8c5bb848
# ╟─82b0a368-90be-11eb-0ddb-310f332a83f0
# ╠═f27bdffa-90c0-11eb-0f71-6d572f799290
# ╠═edef7f53-02a3-4841-a9e3-56706f0e5ef4
# ╠═00b7f55e-90c1-11eb-119e-f577037923a9
# ╠═f41c2f1d-f6f9-49e0-aa20-73b2ce55bf7f
# ╠═cf4d6579-52a5-45bc-81ed-6797f59a2846
# ╟─02181adc-90c1-11eb-29d7-736dce72a0ac
# ╠═3a0e1d5a-90c2-11eb-16a7-8f9de1ea09e4
# ╠═3a9a2624-90c2-11eb-1986-17b80a2a58c5
# ╠═3b6f8e5e-90c2-11eb-3da4-a5fd3048ab63
# ╠═4a43c8cc-b2f6-4364-b5f3-8bd58e3f2f2e
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
# ╠═78714227-094e-4a00-a72c-accfb50e1b71
# ╟─5ca39a1b-4871-4f48-9934-99c88fb504ba
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
# ╠═5926c5c0-c552-4802-8447-58e526e00659
# ╠═2c8ae3f0-9e01-40b3-9cb0-0d651a048662
# ╠═f7e9c8c1-7939-4551-9267-28e58bb63191
# ╠═780d91c4-c606-48d5-bda6-7512f4ab2d82
