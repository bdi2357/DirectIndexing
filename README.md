# Direct Indexing

This repository showcases the capabilities and strategies of **Direct Indexing**, a personalized and efficient investment approach that replicates the performance of an index while allowing for individual customizations.

---

## Objective

The primary objective of this project is to demonstrate and explore **Direct Indexing** through two key scenarios:

1. **Replicating an Index with Known Constituents**  
   Develop an optimizer to closely follow an index (e.g., S&P 500) by minimizing **active risk**.
   - **Active Risk**: Defined as the standard deviation of the difference between the portfolio return and the benchmark return.
   - **Goal**: Achieve low tracking error while maintaining alignment with the benchmark's risk-return profile.

2. **Replicating an Index with Unknown Constituents**  
   Design a trading strategy to mimic the returns of a fund, given:  
   - **Fund Returns**: Historical daily (or other frequency) returns of the target fund.  
   - **Eligible Universe**: A specified domain of securities eligible for investment.  

   For the proof of concept (POC), the **S&P 500** will serve as the index, and a subset of its constituents will define the eligible universe.

---

## Features

- **Portfolio Optimization**: Advanced optimization techniques to minimize tracking error.
- **Return Mimicking**: Strategies to estimate and follow fund returns even when the exact constituents are unknown.
- **Flexibility**: Adaptability to various benchmarks, asset classes, and investment universes.

---

## How to Use

1. **Install Dependencies**  
   Ensure that all required libraries and tools are installed. Refer to the `requirements.txt` file for the list of dependencies.

2. **Run the Optimizer**  
   Use the provided scripts to:
   - Optimize a portfolio to closely track a given index.
   - Mimic fund returns with incomplete constituent data.

3. **Customize Your Universe**  
   Experiment with different universes of eligible securities and observe the impact on tracking error and portfolio performance.

---

## Future Work

- Extend the framework to non-U.S. indices (e.g., MSCI World, FTSE 100).
- Incorporate ESG filters for more tailored portfolio construction.
- Develop a dashboard for real-time performance monitoring and visualizations.

---

## Contributing

We welcome contributions to enhance the project. Please fork the repository, create a new branch, and submit a pull request with your changes.

---

## License

This project is open-source and available under the [MIT License](LICENSE).
