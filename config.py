import os

class Config(object):
    DEBUG = False
    TESTING = False

class ProductionConfig(Config):
    DEBUG = False
    PORT = os.environ.get('PORT', 8080)

class DevelopmentConfig(Config):
    DEBUG = True
    PORT = 8080

class TestingConfig(Config):
    TESTING = True
    PORT = 8080