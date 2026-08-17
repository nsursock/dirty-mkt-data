Excellent project idea, "dirty-mkt-data." It captures the essential challenge of working with financial time series. You're right to start with **GBM** and **GARCH**—they are the absolute foundation, and the "SOTA" landscape has evolved significantly with generative and deep learning approaches. Here's a structured overview of the field to help you plan your project, including the established classics and the latest innovations.

### 1. The Classic Foundation: GBM & GARCH

These are your starting point for a reason. They are the benchmarks against which all new models are compared.

*   **Geometric Brownian Motion (GBM)**: The fundamental model for stock price paths, assuming constant volatility and log-normal returns. It's computationally efficient but fails to capture stylized facts like volatility clustering and heavy tails . For short-term forecasts (e.g., 30 days), it can be surprisingly accurate, but errors grow with the horizon due to its simple structure .
*   **GARCH Family (GARCH, EGARCH, GJR)**: These models are the primary tools for modeling **volatility clustering**—the tendency for high-volatility periods to persist. The standard GARCH(1,1) is a workhorse . For capturing asymmetric effects (where bad news impacts volatility more than good news), you should look into **EGARCH** or **GJR-GARCH**, which often outperform the standard GARCH .
*   **Jump Diffusion**: To address GBM's inability to handle sudden, large price movements (jumps), you can extend it with a **Jump Diffusion** process (like the Merton model). This is crucial for realistic risk management .

**Modern Twist**: A recent study benchmarked forecasting models on simulated SDEs, including GBM and the Heston model (stochastic volatility). Interestingly, **Temporal Convolutional Networks (TCNs) achieved the lowest MSE**, while **LSTMs and Transformers** also performed well, though no model could predict the direction of random noise .

### 2. State-of-the-Art (SOTA) and Advanced Directions

The field is moving beyond simple parametric models to more flexible, data-driven approaches.

*   **Generative Models & Benchmarks**: The state-of-the-art now heavily involves **Generative AI**. Tools like **TimeGAN**, **VAEs**, and **Diffusion Models** are being used to generate highly realistic synthetic time series data . For evaluation, you can leverage new benchmarks designed to stress-test models on specific financial mechanisms.
    *   **FinStressTS**: A diagnostic benchmark with 30 controlled environments to test models on specific challenges like volatility clustering, heavy tails, and regime switching .
    *   **CTBench**: The first benchmark specifically for cryptocurrency time series generation, evaluating models not just on statistical accuracy but also on trading performance and risk assessment .

*   **Deep Learning Architectures**:
    *   **MacroVAE**: This uses a VAE conditioned on macroeconomic variables to generate realistic financial scenarios, including heavy tails and volatility clustering .
    *   **MarketSim & MarS**: These represent the frontier of market simulation. **MarketSim** uses generative agents (powered by LLMs) to simulate interactions between thousands of participants . **MarS** uses a "Large Market Model" to generate realistic, order-level market data . These are for building full-fledged market simulators, not just price path generators.
    *   **JDAPP**: This framework integrates **APARCH** volatility (a GARCH variant), **Jump Diffusion**, and **Fourier-Block Randomization** to create a robust synthetic data generator specifically for training Reinforcement Learning (RL) trading agents .

*   **Causal Approaches**: If you want to truly understand the *why* behind market movements, look into causal models. Recent work on **Causal Market Simulators** uses **Time Causal VAEs** to enforce a causal structure on data generation, allowing for counterfactual analysis (e.g., "what would have happened if the Fed had cut rates?") .

### 3. Suggested Project Roadmap for "dirty-mkt-data"

To build a robust project, consider this phased approach:

1.  **Foundation**: Implement and rigorously test **GBM** and **GARCH(1,1)** on a chosen asset class (e.g., EUR/USD or a major stock index). Validate your models on key "stylized facts" like heavy tails and volatility clustering .
2.  **Advanced Classic**: Extend your GARCH model to an asymmetric version like **EGARCH** or **GJR-GARCH**. Then, add a jump-diffusion component to your GBM to see if it improves simulation quality. Evaluate them using metrics like MSE, MAE, and risk metrics like VaR/CVaR .
3.  **Deep Learning**: Implement a **TCN** or **LSTM** model for forecasting. Compare its performance against your GBM/GARCH benchmark to see if the added complexity pays off . You could also use this step to integrate the **JDAPP** framework for generating synthetic training data to feed into a simple trading agent .
4.  **SOTA Exploration**: This is where you can get creative. Try a model like **TimeGAN** or **MacroVAE** to generate synthetic price paths. If you're ambitious, use a benchmark like **FinStressTS** to systematically evaluate your models' weaknesses .

This roadmap will take you from a solid, replicable foundation to the cutting edge, demonstrating a deep understanding of both classical theory and modern AI applications in finance. Good luck with the project—it's a fantastic area to explore.