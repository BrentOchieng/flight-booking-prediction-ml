#  British Airways – Customer Booking Prediction Project

##  Overview

This project focuses on predicting whether a customer will **complete a flight booking**, using behavioral and trip-related features such as purchase lead time, flight hour, day of flight, length of stay, and sales channel.

Two machine learning models were developed and evaluated:

- **Logistic Regression**
- **Random Forest Classifier**

The goal was to identify the most accurate model for predicting booking completion and provide business recommendations that can improve customer engagement and booking conversions.

---

#  Problem Statement

Many customers abandon the flight booking process before confirming. Predicting which customers are likely to complete their booking allows British Airways to:

- Target high-intent customers with reminders or discounts  
- Optimize marketing and retargeting campaigns  
- Personalize customer engagement  
- Improve overall conversion rates  

This project builds a machine learning classification system to predict whether a customer completes a booking (`booking_complete = 1`) or not (`booking_complete = 0`).

---

#  Data Preprocessing

### Categorical Encoding  
- One-hot encoding for flight day (`Mon–Sun`)
- Sales channels converted to numerical values

### Feature Selection  
Used the following features:

- `length_of_stay`  
- `flight_hour`  
- `purchase_lead`  
- `sales_channel_num`  
- One-hot encoded flight day variables

###  Data Splitting  
- 80% Training  
- 20% Testing  

###  Class Imbalance  
The dataset is skewed with more **non-bookings** than **bookings**, influencing how models were evaluated.

---

#  Models Developed

Two models were implemented and compared.

---

## 1️ Logistic Regression

### **Performance Summary**
| Metric | Value |
|--------|--------|
| Accuracy | ~0.45 |
| Recall (Bookings) | Very low |
| F1 Score | Very low |
| AUC | ~0.50 |

###  Key Issues
- Failed to handle nonlinear relationships  
- High false positives and false negatives  
- Poor prediction of actual bookings (minority class)

 **Conclusion:** Logistic Regression is **not suitable** for this prediction task.

---

## 2️ Random Forest Classifier

### **Performance Summary**
| Metric | Value |
|--------|--------|
| Accuracy | ~0.82 |
| AUC | ~0.57 |
| Recall (Bookings) | Significantly better |
| F1 Score | Improved after threshold tuning |

###  Strengths
- Captures complex nonlinear patterns  
- Handles class imbalance better  
- Higher overall prediction power  
- Allows tuning of probability thresholds  
- Provides interpretable feature importance  

 **Conclusion:** Random Forest is the **best model** and recommended for deployment.

---

#  Final Model Recommendation

##  **Recommended Model: Random Forest Classifier**

The Random Forest Outperformed Logistic Regression across:

- Accuracy  
- Recall  
- F1 Score  
- Prediction stability  
- Interpretability  
- Threshold flexibility  

### Why Random Forest?
- Learns nonlinear behavior of booking patterns  
- More effective for imbalanced datasets  
- Provides more reliable booking predictions  
- Higher recall helps capture actual customers likely to book  

---

# Feature Importance Insights

Top features influencing booking completion:

1. **Purchase Lead**  
2. **Length of Stay**  
3. **Flight Hour**  
4. **Sales Channel**  
5. **Flight Day (one-hot encoded)**  

These insights can inform targeted marketing and customer engagement strategies.

---

#  Evaluation Metrics Used

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- AUC (ROC)  
- Confusion Matrix  
- Classification Report  

---

# Conclusion

This project successfully builds a machine learning model to classify customer booking completion for British Airways. After evaluating both models:

### **The Random Forest Classifier is the best-performing model**  
because it provides higher predictive accuracy, better recall for the booking class, and stronger overall performance.

With additional features, class balancing, and threshold tuning, this model can drive:

- Better customer targeting  
- Higher booking conversions  
- More efficient marketing strategies  
- Improved customer experience  

---

#  Author

**Brent Ochieng**  
Data Analyst | Machine Learning Enthusiast  

---







