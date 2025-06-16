# MAGI & DDE-Find Algorithms: A README for Parameter Inference in Dynamical Systems

This document provides a concise overview of the **Manifold-constrained Gaussian Process Inference (MAGI)** algorithm and its extensions, alongside the related **DDE-Find** framework. It highlights the purpose, mathematical underpinnings, and practical implementations of these methods, developed and analyzed during the first weeks of a research internship at the Institute of Marine Research, Norway. This work focuses on solving inverse problems for differential equations, a critical task in modeling complex systems from sparse and noisy data.

---

## What is MAGI?

**MAGI** is a powerful Bayesian inference framework designed to estimate the parameters of **Ordinary Differential Equations (ODEs)** directly from noisy and often sparse time-series data. The core challenge in this area is to find the values of unknown constants (e.g., reaction rates, growth parameters) in a system of ODEs that best explain the observed measurements.

### The Core Idea

The algorithm, originally proposed by **Yang, Wong, and Kou in their 2021 PNAS paper**, ingeniously combines two powerful mathematical concepts:

* **Gaussian Processes (GPs):** GPs are a non-parametric method used to model functions. In MAGI, a GP is placed as a prior over the latent (unobserved) trajectories of the system's state variables. This allows for a flexible representation of the system's behavior between discrete data points.
* **Manifold Constraint:** The key innovation is the "manifold constraint." This enforces the idea that the true underlying trajectory of the system must satisfy the physics or rules described by the ODEs. MAGI evaluates the likelihood of the GP-represented trajectory by measuring how well its derivatives match the dynamics dictated by the ODE system at every point in time.

By combining these, MAGI formulates a posterior probability distribution over the ODE parameters and the latent system states. It then uses advanced **Markov Chain Monte Carlo (MCMC)** methods, specifically **Hamiltonian Monte Carlo (HMC)** and the **No-U-Turn Sampler (NUTS)**, to draw samples from this posterior. The resulting samples provide not only point estimates for the parameters but also their credible intervals, offering a complete picture of the estimation uncertainty.

### Mathematical Fields Involved

MAGI operates at the intersection of several key mathematical and statistical fields:
* **Bayesian Statistics:** For its inferential framework.
* **Machine Learning:** Utilizing Gaussian Processes.
* **Numerical Analysis & Differential Equations:** For handling the ODE systems.
* **Stochastic Processes & MCMC:** For sampling from the posterior distribution.

---

## Extending MAGI for Delayed Systems: MAGIDDE

Many real-world systems, from biological processes to ecological networks, exhibit time delays. For instance, a predator population's growth might depend on the prey population from a week ago, not the current moment. These are modeled using **Delay Differential Equations (DDEs)**.

Building on the original MAGI, **Zhao and Wong (2024)** developed **MAGIDDE**, an extension specifically for inferring parameters in DDEs. The primary challenge was to handle the time-delayed terms within the manifold constraint without resorting to computationally expensive numerical solvers. MAGIDDE cleverly addresses this by using a linear approximation for the delayed state values, allowing the efficient gradient-based HMC sampling to be preserved.

## Project Implementations and Findings

This research project involved the implementation, testing, and extension of these cutting-edge algorithms, with a focus on applying them to ecological models.

### MAGI.jl: A Julia Implementation

A core part of the project was the development of a **Julia implementation of the MAGI algorithm**. This implementation is now largely feature-complete and was successfully tested on benchmark ODE systems like the FitzHugh-Nagumo model. The results from a progress report show that the implementation can accurately recover true parameter values from simulated noisy data, both when the observation noise is known and when it is inferred as part of the model.

### DDE-Find: An Alternative Approach

In parallel, the **DDE-Find** algorithm, introduced by **Stephany et al. (2024)**, was explored. DDE-Find uses a different philosophy based on **adjoint sensitivity analysis** to efficiently compute gradients for an optimization-based approach. The project leveraged the powerful **SciML ecosystem in Julia** to replace manual adjoint calculations with automatic, robust, and high-performance methods. Initial tests on benchmark DDEs showed promising results, though also highlighted sensitivity to initial parameter guesses.

### Predator-Prey Modeling and Future Work

A key focus of the internship was understanding and modeling a real-world **predator-prey system with time delays**, as described by **Frank et al. (2021)**. The ultimate goal is to apply the developed **MAGIDDE** and **DDE-Find** frameworks to infer the parameters of such ecological models from actual marine research data.

Future directions for this project include:
* Finalizing the **MAGIDDE** implementation in Julia.
* Conducting extensive **benchmarking** to compare the performance, robustness, and scalability of MAGIDDE and DDE-Find.
* Investigating **hybrid methods** that could leverage the strengths of both the Bayesian (MAGI) and optimization-based (DDE-Find) approaches.
* Applying these advanced inference tools to solve real-world challenges in marine science.

This work represents a significant step towards building robust, efficient, and reliable tools for uncovering the hidden dynamics of complex systems from limited data, with direct applications in scientific research and beyond.