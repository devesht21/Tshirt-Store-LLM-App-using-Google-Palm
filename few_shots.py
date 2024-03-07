few_shots = [{'Question': "How many tshirts we have left for Van Huesen in extra large size and black color?", 'SQLQuery': "SELECT SUM(stock_quantity) FROM t_shirts WHERE brand = 'Van Huesen' AND color = 'Black' AND size = 'XL'", 'SQLResult': "Result of SQL Query", 'Answer': "97"},
            {'Question': "How much is the price of the inventory for all small size tshirts?", 'SQLQuery': "SELECT SUM(price*stock_quantity) FROM t_shirts WHERE size = 'S'", 'SQLResult': "Result of SQL Query", 'Answer': "12482"},
            {'Question': "If we have to sell all the Levi's tshirts today with discount applied, How much revenue my store will generate (post discounts)?", 'SQLQuery': "select sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from (select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = 'Levi' group by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id", 'SQLResult': "Result of SQL Query", 'Answer': "26098.35"},
            {'Question': "How many white color Levi's tshirts do we have?", 'SQLQuery': "SELECT SUM(stock_quantity) FROM t_shirts WHERE brand = 'Levi' AND color = 'White'", 'SQLResult': "Result of SQL Query", 'Answer': "125"},
            {'Question': "How many Nike tshirts do we have?", 'SQLQuery': "SELECT SUM(stock_quantity) FROM t_shirts WHERE brand = 'Nike'", 'SQLResult': "Result of SQL Query", 'Answer': "494"},
            {'Question': "Which Tshirt have the maximum discount on it?", 'SQLQuery': "SELECT brand, color, size FROM t_shirts WHERE t_shirt_id = (SELECT t_shirt_id FROM discounts ORDER BY pct_discount DESC LIMIT 1)", 'SQLResult': "Result of SQL Query", 'Answer': "Adidas, Black, L"},
            {'Question': "Which tshirts have more than average discount on it?", 'SQLQuery': "SELECT brand, color, size FROM t_shirts WHERE t_shirt_id = (SELECT t_shirt_id FROM discounts ORDER BY pct_discount DESC LIMIT 1)", 'SQLResult': "Result of SQL Query", 'Answer': "Adidas, Black, L"},]