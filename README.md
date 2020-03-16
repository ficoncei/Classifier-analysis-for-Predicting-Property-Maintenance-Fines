# Classifier-analysis-for-Predicting-Property-Maintenance-Fines

One of the most pressing problems in Detroit is blight. Blight violations are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the question can be posed: how can blight ticket compliance be increased?

The code presented in this repository analyses the available datasets and creates different classifiers to predict the probability that a new fine will be paid or not on time.

The data available in the datasets has been provided through the Detroit Open Data Portal.

Two data files are used in training and validating the models: train.csv and test.csv. Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing data, False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible. A thorough description of the fields in the dataset can be found at the bottom of this readme file.


##Comparing classifiers
The classifiers built with the available data in the train.csv dataset are 'Naive Bayes','Gradient boosting','Random Forest' and 'Tuned Random Forest'. The latter, 'Tuned Random Forest', is obtained after running a Randomised Search for two parameters, 'n_estimators' and 'max_depth'. The search has been limited to two parameters and to a very small subset of possible values due to the computational effort required. Comparison between the classifiers is displayed here:

<p align="center">
  <img src="https://github.com/ficoncei/Classifier-analysis-for-Predicting-Property-Maintenance-Fines/blob/master/files/meanAUC.png">
</p>

An analysis of the data in the train.csv dataset is presented. The bar plots show the influence of each feature on the compliance attribute:

<p align="center">
  <img src="https://github.com/ficoncei/Classifier-analysis-for-Predicting-Property-Maintenance-Fines/blob/master/files/comp_adminfee.png">
</p>

<p align="center">
  <img src="https://github.com/ficoncei/Classifier-analysis-for-Predicting-Property-Maintenance-Fines/blob/master/files/comp_agencyname.png">
</p>

<p align="center">
  <img src="https://github.com/ficoncei/Classifier-analysis-for-Predicting-Property-Maintenance-Fines/blob/master/files/comp_cleanupcost.png">
</p>

<p align="center">
  <img src="https://github.com/ficoncei/Classifier-analysis-for-Predicting-Property-Maintenance-Fines/blob/master/files/comp_discount.png">
</p>

<p align="center">
  <img src="https://github.com/ficoncei/Classifier-analysis-for-Predicting-Property-Maintenance-Fines/blob/master/files/comp_disposition.png">
</p>

<p align="center">
  <img src="https://github.com/ficoncei/Classifier-analysis-for-Predicting-Property-Maintenance-Fines/blob/master/files/comp_fineamount.png">
</p>

<p align="center">
  <img src="https://github.com/ficoncei/Classifier-analysis-for-Predicting-Property-Maintenance-Fines/blob/master/files/comp_hearingtimediff.png">
</p>

<p align="center">
  <img src="https://github.com/ficoncei/Classifier-analysis-for-Predicting-Property-Maintenance-Fines/blob/master/files/comp_inspectorname.png">
</p>

<p align="center">
  <img src="https://github.com/ficoncei/Classifier-analysis-for-Predicting-Property-Maintenance-Fines/blob/master/files/comp_judjmentamount.png">
</p>

<p align="center">
  <img src="https://github.com/ficoncei/Classifier-analysis-for-Predicting-Property-Maintenance-Fines/blob/master/files/comp_latefee.png">
</p>

<p align="center">
  <img src="https://github.com/ficoncei/Classifier-analysis-for-Predicting-Property-Maintenance-Fines/blob/master/files/ 	comp_statefee.png">
</p>

<p align="center">
  <img src="https://github.com/ficoncei/Classifier-analysis-for-Predicting-Property-Maintenance-Fines/blob/master/files/comp_violationcode.png">
</p>

<p align="center">
  <img src="https://github.com/ficoncei/Classifier-analysis-for-Predicting-Property-Maintenance-Fines/blob/master/files/comp_violatorname.png">
</p>

<p align="center">
  <img src="https://github.com/ficoncei/Classifier-analysis-for-Predicting-Property-Maintenance-Fines/blob/master/files/comp_zipcode.png">
</p>

Some of the features display higher variance and that means some influence the compliance attribute more than others. This analysis is confirmed by performing a Principal Component Analysis. First, the data is passed through the method fit of the PCA class and the cumulative explained variance ratio is plotted:

<p align="center">
  <img src="https://github.com/ficoncei/Classifier-analysis-for-Predicting-Property-Maintenance-Fines/blob/master/files/expvariancecum.png">
</p>

The plot shows that a reduced number of features is enough to explain all the variance in the data. This can be seen in more detail by plotting the variance for each feature in the dataset. In the next plot, it is concluded that one feature is a much bigger contributor than all the others:

<p align="center">
  <img src="https://github.com/ficoncei/Classifier-analysis-for-Predicting-Property-Maintenance-Fines/blob/master/files/expvarianceperfeature0.png">
</p>

Eliminating this feature from the plots gives a better understanding of the variance of the next contributor features:

<p align="center">
  <img src="https://github.com/ficoncei/Classifier-analysis-for-Predicting-Property-Maintenance-Fines/blob/master/files/expvarianceperfeature1.png">
</p>

Eliminating more features from the plot allows to better evaluate the influence of the next top three variance contributor features:

<p align="center">
  <img src="https://github.com/ficoncei/Classifier-analysis-for-Predicting-Property-Maintenance-Fines/blob/master/files/expvarianceperfeature2.png">
</p>

<p align="center">
  <img src="https://github.com/ficoncei/Classifier-analysis-for-Predicting-Property-Maintenance-Fines/blob/master/files/expvarianceperfeature3.png">
</p>

<p align="center">
  <img src="https://github.com/ficoncei/Classifier-analysis-for-Predicting-Property-Maintenance-Fines/blob/master/files/expvarianceperfeature4.png">
</p>



## Description of the data fields in the dataset

train.csv & test.csv

ticket_id - unique identifier for tickets
agency_name - Agency that issued the ticket
inspector_name - Name of inspector that issued the ticket
violator_name - Name of the person/organization that the ticket was issued to
violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
ticket_issued_date - Date and time the ticket was issued
hearing_date - Date and time the violator's hearing was scheduled
violation_code, violation_description - Type of violation
disposition - Judgment and judgement type
fine_amount - Violation fine amount, excluding fees
admin_fee - $20 fee assigned to responsible judgments
state_fee - $10 fee assigned to responsible judgments
late_fee - 10% fee assigned to responsible judgments
discount_amount - discount applied, if any
clean_up_cost - DPW clean-up or graffiti removal cost
judgment_amount - Sum of all fines and fees
grafitti_status - Flag for graffiti violations

train.csv only

payment_amount - Amount paid, if any
payment_date - Date payment was made, if it was received
payment_status - Current payment status as of Feb 1 2017
balance_due - Fines and fees still owed
collection_status - Flag for payments in collections
compliance [target variable for prediction] 
 Null = Not responsible
 0 = Responsible, non-compliant
 1 = Responsible, compliant
compliance_detail - More information on why each ticket was marked compliant or non-compliant
