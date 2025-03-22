# FaceAnalytics

## Project Overview
**FaceAnalytics** is an advanced AI-driven system designed to recognize individuals based on their facial features, analyze their purchasing behavior, and provide detailed insights into customer interactions. The system combines facial recognition, vector embeddings, and data analysis to offer a personalized experience for businesses.

## How It Works
1. **Facial Recognition**: The system uses **harmonic vectors** to identify individuals from their facial features. The facial embeddings are compared with stored embeddings in a **FAISS index** and **PostgreSQL database**.
2. **Data Storage**: When a customer's face is recognized, their profile is retrieved from the database, where data such as past purchases, interactions, and other relevant behaviors are stored.
3. **Analytics**: Once the individual is identified, the system analyzes their previous interactions and purchases, providing businesses with meaningful insights into customer preferences, behaviors, and trends.

## Key Features
- **Harmonic Vector Matching**: Uses harmonic vectors to accurately identify individuals based on their facial features.
- **FAISS Indexing**: Efficiently stores and searches facial embeddings using **FAISS**, a powerful library for nearest neighbor search.
- **PostgreSQL Integration**: Securely stores customer profiles, purchase history, and other relevant data in a **PostgreSQL** database.
- **Personalized Analytics**: Analyzes customer data to provide personalized purchasing insights, enhancing customer experience and business strategy.

## Tech Stack
- **Face Recognition**: **FaceNet** for generating facial embeddings.
- **Vector Search**: **FAISS** (Facebook AI Similarity Search) for fast vector matching and retrieval.
- **Database**: **PostgreSQL** for storing user profiles and historical data.
- **AI Framework**: **PyTorch** for deep learning models.

## API Endpoints

### POST /face_recognize
**Description**: Recognizes a customer by their face, retrieves their profile, and provides insights into their purchase behavior.

**Request**:
- `file`: Image with the customer's face.
- `description`: Optional field with a description of the customer.

### POST /add_new_person
**Description**: Adds a new customer to the database by processing their face and storing their profile and purchase history.

**Request**:
- `file`: Image with the customer's face.
- `description`: A description of the new customer (e.g., name, preferences).

## Future Enhancements
- **LLM (Large Language Models) Integration**: Integrating advanced LLMs for more accurate customer behavior predictions, generating personalized content and messages based on past interactions.
- **RECSYS (Recommendation System)**: Implementing a recommendation system to suggest products or services based on customer preferences and purchase history, enhancing the customer experience.
  
Если после создания контейнера не запускается сеть, то:
```
docker network create mynetwork
docker network connect mynetwork backend_service
docker network connect mynetwork frontend_service
```

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Documentation
For full documentation, visit: [FaceAnalytics Documentation](https://nikitamldl.github.io/FaceAnalytics/)
