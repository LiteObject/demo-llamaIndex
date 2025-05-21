CREATE TABLE country_stats (
    country_id SERIAL PRIMARY KEY,
    country_name VARCHAR(50) NOT NULL UNIQUE,
    country_population INTEGER
);

CREATE TABLE city_stats (
    city_id SERIAL PRIMARY KEY,
    city_name VARCHAR(50) NOT NULL,
    city_population INTEGER,
    country_id INTEGER REFERENCES country_stats(country_id)
);

INSERT INTO country_stats (country_name, population) VALUES
('Canada', 38000000),
('Japan', 125800000),
('United States', 331000000),
('South Korea', 51780000),
('United Kingdom', 68200000),
('Australia', 25690000),
('Germany', 83100000),
('South Africa', 59310000);

INSERT INTO city_stats (city_name, population, country_id) VALUES
('Toronto', 2930000, 1),
('Tokyo', 13960000, 2),
('Chicago', 2679000, 3),
('Seoul', 9776000, 4),
('Fargo', 125990, 3),
('Frisco', 207908, 3),
('London', 8982000, 5),
('Sydney', 5312000, 6),
('Dallas', 1343000, 3),
('Berlin', 3769000, 7),
('Cape Town', 4337000, 8);