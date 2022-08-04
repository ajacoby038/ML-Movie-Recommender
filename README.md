# SummerProject

## Intro to Movie Recommendation Systems

### What is a movie recommendation system?
- An **ML-based approach** to filtering or predicting users' film preferences based on their past choices or behavior
- **Advanced filtration** mechanism that predicts possible movie choices

### Architecture
- Two main elements: **users** and **items**
    - generates movie predictions for users
    - items are the movies themselves

### Filtration Techniques
1. **Content based filtering**
- Filtration technique which uses data about the movies watched by a single user
- Recommends movies based on similarity to past watched movies
2. **Collaborative filtering**
- Filtration technique which uses data about the movies watched by multiple users
- Recommends movies based on a collaboration of multiple users' preferences
- Two categories of collaborative filtering algorithms
    1. User-based collaborative filtering
        - Look for similar patterns in movie preferences in the target user and other users
    2. Item-based collaborative filtering
        - Look for similar movies that target users rate/interact with
- **Modern approach uses a mix of both techniques for best results**

### How to build a Movie Recommendation System?
- What is required?
    1. Data
        - ML systems need data, so import movie datasets with ratings
    2. Analysis
        - Create generic recommendations of top-rated movies from the dataset
    3. Personalization
        - Get personalized recommendations by providing your own movie scores
    4. Strategy
        - Implement one of the filtration techniques
    5. Combination
        - Combine recommendation lists to get a reasonable estimate accross ratings
- How to create a neural network model?
    - Artificial neural networks (ANNs)
        - Use training data to predict movie recommendations with high accuracy ofr the target user
        - Most important part is to get the right movie datasets and make the right manipulations with the data
    - 3 layers of a neural network model
        1. Input
            - First layer where the movie and user vectors are selected as input
        2. Embedding
            - Second layer has embeddings for both movies and users
            - Updated during model training to get best values of these embeddings and lower error rate
        3. Output
            - Third layer generates tha predicted values
- Once ANN is created, train it with the movie dataset and make predictions