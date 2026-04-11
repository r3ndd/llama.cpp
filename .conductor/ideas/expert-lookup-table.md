# Expert Lookup Table

I want to test a hypothesis I have around a potential method to make LLMs easier to run on local hardware, specifically LLMs of the mixture-of-experts type.
My core idea is to make a modification to llama.cpp that allows me to algorithmically reduce the number of experts in a MoE LLM by replacing their computation with static lookup lookup vectors. This would be broken down into two steps: **training** and **inference**.

## Training
In "training", I would run inference on the model for a variety of prompts, tracing which experts are routed to in various contexts. Then, after ranking each expert (within each layer) by some heuristic (least used, most redundant, etc.), I would select a specific percentile to replace.
To replace an expert, there are two algorithms I am considering. In both cases, the input to the top-k experts would be mapped to a discrete set of keys per layer using k-means clustering, and then the outputs would be averaged in a lookup table at those keys.

### Algorithm 1: Unique Table per Expert
In this algorithm, each removed expert would have its own lookup table, with its own unique keys derived through k-means clustering from when that expert is in the top-k for a token. Since having a lookup table for every expert would consume a lot of storage, experts would need to be trained in small batches and their final saved output vectors would be binary vectors of the average (after Hadamard transform).

### Algorithm 2: Shared Table per Layer
In this algorithm, all experts within a layer would share the same lookup table, with keys derived in the same way as in Algorithm 1 but with them being the same for all experts.
However, the output of the table would be the weighted, combined output of the removed top-k experts.


## Inference
During inference, the modified llama.cpp would use the lookup tables to replace the computation of the selected experts.
For Algorithm 1, the top-k experts that were not replaced would be computed as usual (including the next best not replaced experts for k total computed), while any replaced top-k experts would have their output computed by looking up the corresponding key in their unique lookup table, doing the inverse Hadamard transform, and then combining all expert outputs as usual (weighted by router's scores).
For Algorithm 2, the top-k experts that were not replaced would be computed as usual (along with the next best not replaced experts for k total computed), while any replaced top-k experts would have their output determined by looking up the corresponding key in the shared lookup table and then adding it to the combined output of the non-replaced experts. The computed experts' outputs would be weighted by the router's scores as usual, and the loaded output would be weighted by the router's net score for the missing top-k experts.

## Conclusion
The core advantage of this approach, if it works, is that it would allow for the parameters of large models to be loaded entirely into memory, while still preserving learned expert knowledge and routing behavior. Increasing the size of the lookup tables would improve the performance at a given number of replaced experts, while only increasing storage requirements and not affecting inference time (constant time lookups). Since storage is cheaper than both memory and compute, this could be a good tradeoff.
The key parameters and implementation details that I would need to experiment with include:
- The heuristic for ranking experts and percentage of experts to replace
- The number of keys to use in the lookup table (probably 10,000+ to start)
- Training/lookup algorithm (Algorithm 1 vs Algorithm 2)
