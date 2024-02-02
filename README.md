# Anime Recommender System

Welcome to the Anime Recommender System project! 
This project aims to provide personalized recommendations for anime series based on user preferences.

## Introduction

This repository contains the source code and data for an anime recommender system. The system leverages a collaborative filtering algorithm to recommend anime series to users based on their interests and the similarities between different anime titles. 
The recommendations take into account factors such as user preferences, anime summaries, and tags associated with each series.

## Installation

To use this recommender system, follow the steps below:


1. Clone the repository to your local machine.
2. Install the required dependencies listed in the requirement.txt file.
3. Run command streamlit run app.py file to start the application.

## Usage

Once the system is running, follow these steps to get anime recommendations:


1. Select an anime series from the dropdown menu.
2. Click the "Recommend" button.
3. The system will display the top recommendations along with their titles, summaries, and tags.

## Data

The anime data used in this system includes information about various series, such as their names, synopses, and tags. 
The data is stored in a pickle file format and is loaded into the system at runtime.

## Recommendation Algorithm

This system employs a collaborative filtering algorithm to generate anime recommendations. 
It calculates the similarity between different anime series based on user preferences and suggests series that are similar to the selected anime. 
The recommendations are sorted based on their similarity scores.

## Dependencies

The following libraries are required to run the system:

- streamlit
- numpy
- pandas
- pickle

## Contributing

Contributions to this project are welcome. If you find any bugs, have suggestions for enhancements, or would like to contribute in any way, please feel free to open an issue or submit a pull request.
