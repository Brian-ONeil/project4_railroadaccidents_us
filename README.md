# Human Factor Caused Railroad Accidents in the US

## Project Description
* This project looks at railroad accidents data between 2019 to present from the Federal Railroad Administration (FRA) in search of relationships among the features with a target of "cause" (specifically caused by human factor "H"). We will compare subpopulations of human actor accidents versus all others using different features. It's the intent of the project, that it will produce a model that will help FRA intervene and decrease human related railroad accidents.

## Project Goals

* Find potential drivers for human factor caused railroad accidents in the US

* Produce a viable model to predict high risk factors contributing to human factor related railroad accidents.

* Propose actionable options for the FRA to change policy for railroad companies to further prevent human factor related railroad accidents.

* Other key drivers:
    * Weather
    * Visibility
    * Conductor Hours
    * Train Speed
## Initial Thoughts
* There appears to be some key indicators in the data that may lead to human related caused accidents and it is my intent to reveal those indicators by the conclusion of the project.

## The Plan

* Acquire and join .csv's containing all 2019 to present railroad accidents in the US.

* Prepare the data using the following columns:
    * target: 'cause' (human factor-subpopulation)
    * features used:
        * Weather
        * Visibility
        * Conductor Hours
        * Train Speed

* Explore dataset for predictors of cause
    * Answer the following questions:
        * Does visibility affect human factor caused accidents?
        * Does weather affect human factor caused accidents?
        * Does conductor work hours affect human factor caused accidents?
        * Does train speed affect human factor caused accidents?

* Develop a model
    * Using the selected data features develop appropriate predictive models
    * Evaluate the models in action using train and validate splits
    * Choose the most accurate model 
    * Evaluate the most accurate model using the final test data set
    * Draw conclusions

## Data Dictionary
| Feature          | Definition                                                                       |
|------------------|----------------------------------------------------------------------------------|
| 'date'           | Year, Month, and Day                                                             |
| 'timehr'         | hour of the day (1-12)                                                           |
| 'timemin'        | minute of the hour (0-59)                                                        |
| 'ampm'           | AM or PM indicator                                                               |
| 'type'           | type of train (e.g. freight, passenger, work)                                    |
| 'state'          | state where the incident occured                                                 |
| 'temp'           | temperature in degrees fahrenheight                                              |
| 'visibility'     | visibility measured in miles                                                     |
| 'weather'        | weather conditions (e.g. clear/PC, rain, fog, snow                               |
| 'trnspd'         | train speed in miles per hour                                                    |
| 'tons'           | train weight in tons                                                             |
| 'loadf1'         | load factor (ratio of weight of train to max allowable weight)                   |
| 'emptyf1'        | empty factor (ratio of weight of train to min allowable weight)                  |
| 'cause' = target | cause of incident (e.g. human error, electrical/mechanical, signal, track, misc) |
| 'acctrk'         | contributing factors related to the track                                        |
| 'actrkcl'        | accident classification (e.g. derailment, collision)                             |
| 'enghr'          | number of hours worked by the engineer at the time of the incident               |
| 'condrhr'        | number of hours worked by the conductor at the time of the incident              |

## Steps to Reproduce
1) Clone the repo: git@github.com:Brian-ONeil/project4_railroadaccidents_us.git in terminal
2) All data is already loaded in .csv format in the directory
2) Using random state 123 
3) Run final_report notebook

## Takeaways and Conclusions
* Exploration: 
    * The statistical modeling showed some relationships with weather and visibility when compared to a subpopulation of human factor caused accidents. 
    * Both p-values for weather and visibility proved a relationship to the human factor cause when compared to against all other cause categories.
    * Statistical modeling when comparing the means of conductor hours with human factor causes and conductor hours with other accident causes did show significance.
    * We also found significance when comparing means of train speed with human factor caused accidents and other accidents.
    * Weather, visibility, conductor hours, and train speed were sent forward for modeling.
* Modeling:
    * The final Random Forest Model on the test data set decreased accuracy, but beat the baseline by almost 3%.
    * It is my opinion there needs to be more viable features added to the models to find any worthy prediction accuracy.

## Recommendations
* The Federal Railroad Administration should consider revamping their accident report forms and policy to include more questions that affect human and job performance on their operators such as:
    * hours of sleep
    * days off
    * training certifications (incomplete or expired)
    * experience level
    * time in that position (new or senior)
* It is important to remember that human factors are the leading railroad accident cause that accounts for ~40% of all accidents. There is always room for improvement when it comes to human related accidents and it's a worthy investment to explore further the causes.

## Next Steps
* Consider bringing in more features such as feature engineering timedate and binning the different shifts into subpopulations.
* Also consider the day of the week the accidents happen such as a weekday versus weekends.