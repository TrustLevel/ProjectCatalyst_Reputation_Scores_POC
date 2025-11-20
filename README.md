# Reputation Scores for Project Catalyst

- Proposal Link: 
https://projectcatalyst.io/funds/11/cardano-use-cases-concept/reputation-scores-for-catalyst-proposers-and-reviewers-by-lidonation-and-trustlevel


# Project Summary

- The goal of this project was to explore how Project Catalyst review data can be used to build a reliable and data-driven reputation system for proposers and reviewers and to demonstrate what an improved review framework could look like. 
- Over the course of Fund 11–13, we designed, built, and refined a Reputation & Expertise (REX) Framework, which mathematically models reviewer reliability and expertise as probabilistic variables. 
- The concept was then validated through TrustLevel’s Fund 14 Review Tool, allowing us to test the REX methodology with real proposals in a live environment. 
- This project thus evolved from a theoretical study of historical data to a practical, probabilistic reputation system that enables fairer, more transparent, and more accountable decision-making in decentralized governance.


# Milestone 1: Research and Data Analysis

Full Report here: https://docs.google.com/document/d/1zSbcMgSx68jl1zpH9wgluEcpJaoGvVDCrRCakF9rzT8/edit?usp=sharing

## 1. Comparative Analysis Linking Reviews with Voting Results and Proposal Outcomes

### Key Findings
- **Fund 9:**
  - Correlation coefficient: 0.045 (very weak positive correlation).
- **Fund 8:**
  - Correlation coefficient: 0.06 (weak positive correlation).
- **Fund 10:**
  - Preliminary data shows a slightly negative correlation.

### Reviewer Accuracy
- Reviewers with an accuracy of 0.9 or higher showed significant correlation with project outcomes, making up about one-third of all reviewers.

### Implications
- Feasibility assessments alone are not sufficient to predict project success.
- A comprehensive evaluation system including additional metrics such as impact assessments is needed.

### Data Overview
- Public Spreadsheets:
  - vCA Aggregated - Fund 8
  - vPA Aggregate File - Fund 9
  - F10 Community Review - Aggregate File 
  - Catalyst Public Reporting Tracker

### Analysis Steps
1. Categorize the Feasibility Rating as:
   - Accurate Rating: Ratings of 4 and 5
   - Wrong Rating: Ratings of 1, 2, and 3
2. Merge data on Project Name to align outcomes with feasibility ratings.
3. Convert Project Status to binary format:
   - Completed = 1
   - Pending = 0
4. Determine correct predictions based on feasibility ratings.
5. Calculate the percentage of outcomes correctly predicted by each reviewer.
6. Create box plots to compare feasibility ratings by project outcome.

### Fund Analysis
- **Fund 8:**
  - 77.46% of reviewers predicted outcomes correctly at least 50% of the time.
  - Correlation coefficient: 0.0685 (very weak positive correlation).
- **Fund 9:**
  - Overall correlation: 0.0454 (very weak positive correlation).
  - For reviewers with high accuracy (≥ 0.9), correlation: ~0.499.
  - Mean accuracy: 70.19%.
- **Fund 10:**
  - Correlation with projects being completed on time: -0.046.
  - Correlation with projects being completed: -0.137.

## 2. Development of a Knowledge Graph and Network Analysis to Identify Quality Reviewers

### Ontology Structure
- **Entities (Nodes):**
  - Project: Attributes (`Project Title`, `Overall Feasibility Rating`, `Project Status`)
  - Reviewer: Attributes (`Reviewer ID`, `Reviewer Accuracy`)
- **Relationships (Edges):**
  - Reviewed: Connects a `Reviewer` to a `Project` with attributes (`Feasibility Rating`, `Rating Accuracy`).

### Steps to Create the Knowledge Graph
1. Create Nodes for projects and reviewers.
2. Create Edges connecting reviewers to projects reviewed.

### CSV File Structure (see 'data')
- **Project Nodes:**
  - `Project Title`, `Overall Feasibility Rating`, `Project Status`
- **Reviewer Nodes:**
  - `Reviewer ID`, `Reviewer Accuracy`
- **Reviewed Edges:**
  - `Reviewer ID`, `Project Title`, `Feasibility Rating`, `Rating Accuracy`

### Demonstration
- [Knowledge Graph Video](https://www.loom.com/share/43110018ae364a75b0b53fd1d49627f6)

### Network Analysis Insights
- High-Accuracy Reviewers: Identified reviewers with consistent high accuracy.
- Aggregation Across Rounds: Cross-fund evaluation helps establish long-term accuracy scores.

# Milestone 2: POC Develoment and Integration

1. Full POC Framework document here: https://docs.google.com/document/d/1zSbcMgSx68jl1zpH9wgluEcpJaoGvVDCrRCakF9rzT8/edit?usp=sharing

2. Integration of reputation scores and batches into the Catalyst Explorer (v2):
    - Reviewer Profile Page: https://www.catalystexplorer.com/en/ideascale-profiles/g1wbp3jy1r/reviews
    - Proposal Review Overview Page: https://www.catalystexplorer.com/en/reviews
  
# Milestone 3: Community Feedback and Advanced Reputation Framework:

Full Report, including the mathematical foundation of the new REX framework: https://docs.google.com/document/d/1XneKnNcm717duYoGIUvw0iW16z2lXb2Ekq2VYV9-jLY/edit?usp=sharing

  
# Acknowledgments
Special thanks to all contributors and the Catalyst community for their valuable data and insights.
