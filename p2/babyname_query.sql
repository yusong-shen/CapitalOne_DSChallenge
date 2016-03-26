
-- 2.  What is the most popular name of all time? (Of either gender.)
SELECT sex, name, sum(occurence)
FROM namesByState
GROUP BY sex, name
  HAVING sex = 'M'
ORDER BY sum(occurence) DESC
LIMIT 5;

SELECT sex, name, sum(occurence)
FROM namesByState
GROUP BY sex, name
  HAVING sex = 'F'
ORDER BY sum(occurence) DESC
LIMIT 5;

-- 3.  What is the most gender ambiguous name in 2013? 1945?
CREATE VIEW female_1945 AS
SELECT  *
FROM namesByState
WHERE year = 1945 AND sex = 'F';

CREATE VIEW male_1945 AS
SELECT  *
FROM namesByState
WHERE year = 1945 AND sex = 'M';

SELECT DISTINCT f.name AS name, f.occurence as occurence,
  abs(f.occurence - m.occurence) AS difference
FROM female_1945 f, male_1945 m
WHERE f.name = m.name
ORDER BY  difference ASC , f.occurence DESC
LIMIT 5;


CREATE VIEW female_2013 AS
SELECT  *
FROM namesByState
WHERE year = 2013 AND sex = 'F';

CREATE VIEW male_2013 AS
SELECT  *
FROM namesByState
WHERE year = 2013 AND sex = 'M';

SELECT DISTINCT f.name AS name, f.occurence as occurence,
  abs(f.occurence - m.occurence) AS difference
FROM female_2013 f, male_2013 m
WHERE f.name = m.name
ORDER BY  difference ASC , f.occurence DESC
LIMIT 5;


-- 4.  Of the names represented in the data, find the name that has had the largest
--     percentage increase in popularity since 1980. Largest decrease?

CREATE VIEW names_since_1980 AS
  SELECT *
  FROM namesByState
  WHERE year >= 1980;

SELECT count(DISTINCT name)
FROM namesByState
WHERE year = 2014;


-- 5.  Can you identify names that may have had an even larger increase or decrease
--     in popularity?