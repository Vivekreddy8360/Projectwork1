import sqlite3

def create_db():
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('interview_questions.db')
    c = conn.cursor()

    # Create table for questions
    c.execute('''
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stream TEXT NOT NULL,
            level TEXT NOT NULL,
            question TEXT NOT NULL,
            expected_answer TEXT NOT NULL
        )
    ''')

    # Insert questions into the table for multiple streams
    questions_data = [
        # Data Science - Beginner
        ('Data Science', 'Beginner', 'What is the difference between descriptive and inferential statistics?', 'Descriptive statistics summarize data, while inferential statistics use data to make predictions.'),
        ('Data Science', 'Beginner', 'What is a normal distribution?', 'A normal distribution is a bell-shaped curve where most observations cluster around the central peak.'),
        
        # Data Science - Intermediate
        ('Data Science', 'Intermediate', 'What is regularization, and why is it useful?', 'Regularization prevents overfitting by adding a penalty to the model complexity.'),
        ('Data Science', 'Intermediate', 'What is PCA, and how is it used?', 'PCA is a dimensionality reduction technique that projects data into a lower-dimensional space.'),
        
        # Data Science - Advanced
        ('Data Science', 'Advanced', 'What is a hidden Markov model (HMM), and where is it used?', 'A hidden Markov model is a statistical model used in speech recognition and time series analysis.'),
        ('Data Science', 'Advanced', 'How can you avoid overfitting your model?','Keeping the model simple by decreasing the model complexity,Using cross-validation techniques')
        # Machine Learning - Beginner
        ('Machine Learning', 'Beginner', 'What is supervised learning, and how does it differ from unsupervised learning?', 'Supervised learning uses labeled data, while unsupervised learning finds patterns in unlabeled data.'),
        ('Machine Learning', 'Beginner', 'What is overfitting, and how can you prevent it?', 'Overfitting occurs when a model learns noise. It can be prevented by cross-validation or regularization.'),
        
        # Machine Learning - Intermediate
        ('Machine Learning', 'Intermediate', 'What is regularization, and what are L1 and L2 regularization?', 'L1 regularization promotes sparsity, while L2 prevents large weights by adding penalties to the model.'),
        ('Machine Learning', 'Intermediate', 'Explain the difference between bagging and boosting.', 'Bagging trains models independently and combines results, while boosting trains models sequentially, correcting errors from the previous models.'),

        # Machine Learning - Advanced
        ('Machine Learning', 'Advanced', 'What is a convolutional neural network (CNN), and how does it work?', 'A CNN uses convolutional layers to extract features from image data for tasks like classification.'),

        # Python - Beginner
        ('Python', 'Beginner', 'What is a Python list and how do you create one?', 'A Python list is a mutable, ordered collection of elements, created using square brackets [].'),
        ('Python', 'Beginner', 'Explain how you would handle exceptions in Python.', 'Exceptions in Python are handled using try, except, and finally blocks.'),

        # Python - Intermediate
        ('Python', 'Intermediate', 'What are Python decorators, and how do they work?', 'Decorators modify the behavior of a function or method without changing its code by wrapping it.'),
        ('Python', 'Intermediate', 'Explain the Global Interpreter Lock (GIL) in Python.', 'The GIL allows only one thread to execute at a time, preventing true multithreading in CPython.'),

        # Python - Advanced
        ('Python', 'Advanced', 'Explain metaclasses in Python and their use cases.', 'Metaclasses define how classes behave. They allow control over class creation.'),

        # Java - Beginner
        ('Java', 'Beginner', 'What is inheritance in Java?', 'Inheritance allows one class to inherit properties and behaviors from another class using the extends keyword.'),
        ('Java', 'Beginner', 'What is a constructor in Java?', 'A constructor is a special method that is called when an object is instantiated.'),

        # Java - Intermediate
        ('Java', 'Intermediate', 'What is the difference between an abstract class and an interface in Java?', 'An abstract class can have both abstract and non-abstract methods, while an interface can only have abstract methods.'),
        ('Java', 'Intermediate', 'Explain the concept of multithreading in Java.', 'Multithreading in Java allows concurrent execution of two or more threads for maximum CPU utilization.'),

        # Java - Advanced
        ('Java', 'Advanced', 'What is the Java Memory Model (JMM)?', 'The Java Memory Model defines how threads interact with memory and ensures visibility of shared variables.'),

        # C++ - Beginner
        ('C++', 'Beginner', 'What is the difference between a pointer and a reference in C++?', 'A pointer holds the memory address of another variable, while a reference is an alias to an existing variable.'),
        ('C++', 'Beginner', 'What is a constructor in C++?', 'A constructor is a special function that initializes an object when it is created.'),

        # C++ - Intermediate
        ('C++', 'Intermediate', 'What is RAII in C++?', 'RAII stands for Resource Acquisition Is Initialization, a programming idiom for managing resource allocation and deallocation.'),
        ('C++', 'Intermediate', 'Explain the concept of operator overloading in C++.', 'Operator overloading allows you to redefine how operators work with user-defined types in C++.'),

        # C++ - Advanced
        ('C++', 'Advanced', 'What are smart pointers in C++?', 'Smart pointers manage the lifetime of dynamically allocated objects in C++ and ensure proper memory management.')
    ]

    # Insert data into the questions table
    c.executemany('''
        INSERT INTO questions (stream, level, question, expected_answer)
        VALUES (?, ?, ?, ?)
    ''', questions_data)

    conn.commit()
    conn.close()

    print("Database created and data inserted successfully!")

# Run the function to create the database and insert data
create_db()
