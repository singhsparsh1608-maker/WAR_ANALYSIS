SELECT * FROM crea_of_pyhton_sql_ready;

SELECT
month,
AVG(personnel)
FROM crea_of_pyhton_sql_ready
GROUP BY month
ORDER BY AVG(personnel) DESC;

SELECT
SUM(tank) AS total_tanks,
SUM(aircraft) AS total_aircraft,
SUM(drone) AS total_drones
FROM crea_of_pyhton_sql_ready;



