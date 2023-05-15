import java.sql.*;

public class Transform {
    static final String JDBC_DRIVER = "com.mysql.cj.jdbc.Driver";
    static final String DB_URL = "jdbc:mysql://127.0.0.1:3306/sparsedb?allowPublicKeyRetrieval=true&useUnicode=true&characterEncoding=UTF8&useSSL=false&serverTimezone=UTC";
    static final String USER = "root";
    static final String PASS = "123123";

    /**
     * 向sc表中插入数据
     *
     */

    public static void main(String[] args) throws Exception 
    {
        Class.forName(JDBC_DRIVER);
        Connection connection = DriverManager.getConnection(DB_URL, USER, PASS);
        String[] subject = {"chinese", "math", "english", "physics", "chemistry", "biology",
         "history", "geography", "politics"};
        try 
        {
            ResultSet res = connection.createStatement().executeQuery("select * from entrance_exam");
            while (res.next()) 
            {
                int sno = res.getInt("sno"), score;
                for (String sub : subject) {
                    score = res.getInt(sub);
                    if (!res.wasNull())
                    {
                        String sql = "insert into sc values (?, ?, ?) ";
                        PreparedStatement ps = connection.prepareStatement(sql);
                        ps.setInt(1, sno);
                        ps.setString(2, sub);
                        ps.setInt(3, score);
                        ps.executeUpdate();
                    }
                }
            }
        } 
        catch (SQLException e) 
        {
            e.printStackTrace();
        } 
    }
}