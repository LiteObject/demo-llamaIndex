CREATE TABLE city_stats (
    city_name VARCHAR(50) PRIMARY KEY,
    population INTEGER,
    country VARCHAR(50) NOT NULL
);

INSERT INTO city_stats (city_name, population, country) VALUES
('Toronto', 2930000, 'Canada'),
('Tokyo', 13960000, 'Japan'),
('Chicago', 2679000, 'United States'),
('Seoul', 9776000, 'South Korea'),
('Fargo', 125990, 'United States'),
('Frisco', 207908, 'United States');