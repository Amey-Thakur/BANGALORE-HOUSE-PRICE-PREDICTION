<div align="center">

  <a name="readme-top"></a>
  # Bangalore House Price Prediction
  
  [![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
  ![Status](https://img.shields.io/badge/Status-Completed-success)
  [![Technology](https://img.shields.io/badge/Technology-Python%20%7C%20Machine%20Learning-blueviolet)](https://github.com/Amey-Thakur/BANGALORE-HOUSE-PRICE-PREDICTION)
  [![Developed by Amey Thakur & Mega Satish](https://img.shields.io/badge/Developed%20by-Amey%20Thakur%20%26%20Mega%20Satish-blue.svg)](https://github.com/Amey-Thakur/BANGALORE-HOUSE-PRICE-PREDICTION)

  A machine learning study demonstrating the application of **Multivariate Regression** algorithms to estimate real estate prices with high precision based on structural parameters.
  
  **[Source Code](https://github.com/Amey-Thakur/BANGALORE-HOUSE-PRICE-PREDICTION/tree/main/Source%20Code)** &nbsp;·&nbsp; **[Project Report](https://github.com/Amey-Thakur/BANGALORE-HOUSE-PRICE-PREDICTION/blob/main/IIT%20ROPAR%20-%20Diginique%20Techlabs/Project%20Report.pdf)** &nbsp;·&nbsp; **[Live Demo](http://www.diginique.com/)**

</div>

---

<div align="center">

  [Authors](#authors) &nbsp;·&nbsp; [Overview](#overview) &nbsp;·&nbsp; [Features](#features) &nbsp;·&nbsp; [Structure](#project-structure) &nbsp;·&nbsp; [Quick Start](#quick-start) &nbsp;·&nbsp; [License](#license) &nbsp;·&nbsp; [About](#about-this-repository) &nbsp;·&nbsp; [Acknowledgments](#acknowledgments)

</div>

---

<!-- AUTHORS -->
<div align="center">

  <a name="authors"></a>
  ## Authors

  | <a href="https://github.com/Amey-Thakur"><img src="https://github.com/Amey-Thakur.png" width="150" height="150" alt="Amey Thakur"></a><br>[**Amey Thakur**](https://github.com/Amey-Thakur)<br><br>[![ORCID](https://img.shields.io/badge/ORCID-0000--0001--5644--1575-green.svg)](https://orcid.org/0000-0001-5644-1575) | <a href="https://github.com/msatmod"><img src="Mega/Mega.png" width="150" height="150" alt="Mega Satish"></a><br>[**Mega Satish**](https://github.com/msatmod)<br><br>[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--1844--9557-green.svg)](https://orcid.org/0000-0002-1844-9557) |
  | :---: | :---: |

</div>

---

> [!IMPORTANT]
> ### 🤝🏻 Special Acknowledgement
> > *Special thanks to **[Mega Satish](https://github.com/msatmod)** for her meaningful contributions, guidance, and support that helped shape this work.*

---

<!-- OVERVIEW -->
<a name="overview"></a>
## Overview

**Bangalore House Price Prediction** is a machine learning study conducted as part of the **Summer Internship** at **IIT ROPAR - Diginique Techlabs**. The project focuses on the development of a robust regression model capable of predicting property prices in Bangalore with high accuracy.

By leveraging **Scikit-learn**, the system models the real estate landscape where an algorithm learns the relationship between independent variables (Location, Sqft, BHK) and the dependent variable (Price). The model is served via a **Flask** web server for real-time estimation.

### Computational Objectives
The analysis is governed by strict **exploratory and modeling principles** ensuring algorithmic validity:
*   **Dimensionality Reduction**: Handling high-cardinality categorical data (Location) to improve model performance.
*   **Outlier Detection**: Statistical removal of anomalies to ensure robust training boundaries.
*   **Model Selection**: Comparative analysis of Lasso, Ridge, and Linear Regression to minimize Root Mean Squared Error (RMSE).

---

<!-- FEATURES -->
<a name="features"></a>
## Features

| Component | Technical Description |
|-----------|-----------------------|
| **Data Cleaning** | Automated pipeline for handling missing values and removing logical inconsistencies. |
| **Feature Engineering** | Transformation of categorical features using One-Hot Encoding and reduction of rare labels. |
| **Regression Logic** | Implementation of **Linear/Lasso/Ridge** algorithms for price approximation. |
| **Model Persistence** | Serialization of the trained model using **Pickle** for deployment. |
| **Web Interface** | User-friendly UI built with **HTML/CSS/JS** and served via **Flask**. |

> [!NOTE]
> ### Empirical Context
> The real estate dataset consists of diverse property attributes. The inherent variance in property valuations justifies the selection of a robust regression approach over simple linear models, adhering to the requirement for accurate price estimation in a dynamic market.

### Tech Stack
-   **Runtime**: Python 3.x
-   **Machine Learning**: Scikit-learn
-   **Data Manipulation**: Pandas, NumPy
-   **Visualization**: Matplotlib
-   **Web Framework**: Flask

---

<!-- STRUCTURE -->
<a name="project-structure"></a>
## Project Structure

```python
BANGALORE-HOUSE-PRICE-PREDICTION/
│
├── docs/                                            # Technical Documentation
│   └── SPECIFICATION.md                             # Architecture & Design Specification
│
├── IIT ROPAR - Diginique Techlabs/                  # Internship Artifacts
│   ├── Project Report.pdf                           # Final Project Report
│   └── ...                                          # Internship Completion Documents
│
├── Mega/                                            # Collaborative Workspace
│   └── ...                                          # Research & Development resources
│
├── Source Code/                                     # Core Implementation
│   ├── client/                                      # Frontend UI (HTML/CSS/JS)
│   ├── server/                                      # Backend API (Flask)
│   └── model/                                       # Trained Model (Pickle)
│
├── .gitattributes                                   # Git configuration
├── .gitignore                                       # Repository Filters
├── CITATION.cff                                     # Scholarly Citation Metadata
├── codemeta.json                                    # Machine-Readable Project Metadata
├── LICENSE                                          # MIT License Terms
├── README.md                                        # Project Documentation
└── SECURITY.md                                      # Security Policy
```

---

<!-- QUICK START -->
<a name="quick-start"></a>
## Quick Start

### 1. Prerequisites
-   **Python 3.7+**: Required for runtime execution. [Download Python](https://www.python.org/downloads/)
-   **Jupyter Environment**: For interactive model training.

> [!WARNING]
> **Data Integrity**
>
> The prediction kernel depends on specific feature ordering. Ensure that the input vector passed to the model matches the feature columns generated during training (One-Hot Encoded locations).

### 2. Installation
Establish the local environment by cloning the repository and installing the computational stack:

```bash
# Clone the repository
git clone https://github.com/Amey-Thakur/BANGALORE-HOUSE-PRICE-PREDICTION.git
cd BANGALORE-HOUSE-PRICE-PREDICTION

# Install dependencies
pip install pandas numpy matplotlib scikit-learn flask
```

### 3. Execution
Launch the web server to start the prediction application:
```bash
python server/server.py
```

---

<!-- LICENSE -->
<a name="license"></a>
## License

This academic submission, developed for the **Summer Internship** at **IIT ROPAR - Diginique Techlabs**, is made available under the **MIT License**. See the [LICENSE](LICENSE) file for complete terms.

> [!NOTE]
> **Summary**: You are free to share and adapt this content for any purpose, even commercially, as long as you provide appropriate attribution to the original authors.

**Copyright (C) 2021 Amey Thakur & Mega Satish**

---

<!-- ABOUT -->
<a name="about-this-repository"></a>
## About This Repository

**Created & Maintained by**: [Amey Thakur](https://github.com/Amey-Thakur) & [Mega Satish](https://github.com/msatmod)  
**Role**: Summer Interns  
**Program**: Summer Internship  
**Organization**: [IIT ROPAR - Diginique Techlabs](http://www.diginique.com/)

This project features **Bangalore House Price Prediction**, a machine learning study conducted as part of an industrial internship. It explores the practical application of regression analysis in real estate economics.

**Connect:** [GitHub](https://github.com/Amey-Thakur) &nbsp;·&nbsp; [LinkedIn](https://www.linkedin.com/in/amey-thakur) &nbsp;·&nbsp; [ORCID](https://orcid.org/0000-0001-5644-1575)

### Acknowledgments

Grateful acknowledgment to [**Mega Satish**](https://github.com/msatmod) for her exceptional collaboration and scholarly partnership during the execution of this machine learning internship task. Her analytical precision, deep understanding of statistical modeling, and constant support were instrumental in refining the predictive algorithms used in this study. Working alongside her was a transformative experience; her thoughtful approach to problem-solving and steady encouragement turned complex regression challenges into meaningful learning moments. This work reflects the growth and insights gained from our side-by-side academic journey. Thank you, Mega, for everything you shared and taught along the way.

Special thanks to the **mentors at IIT ROPAR - Diginique Techlabs** for providing this platform for rapid skill development and industrial exposure.

---

<div align="center">

  [↑ Back to Top](#readme-top)

  [Authors](#authors) &nbsp;·&nbsp; [Overview](#overview) &nbsp;·&nbsp; [Features](#features) &nbsp;·&nbsp; [Structure](#project-structure) &nbsp;·&nbsp; [Quick Start](#quick-start) &nbsp;·&nbsp; [License](#license) &nbsp;·&nbsp; [About](#about-this-repository) &nbsp;·&nbsp; [Acknowledgments](#acknowledgments)

  <br>

  📈 **[BANGALORE-HOUSE-PRICE-PREDICTION](https://github.com/Amey-Thakur/BANGALORE-HOUSE-PRICE-PREDICTION)**

  ---

  ### Presented as part of the Summer Internship @ IIT ROPAR - Diginique Techlabs

  ---

  ### 🎓 [Computer Engineering Repository](https://github.com/Amey-Thakur/COMPUTER-ENGINEERING)

  **Computer Engineering (B.E.) - University of Mumbai**

  *Semester-wise curriculum, laboratories, projects, and academic notes.*

</div>
