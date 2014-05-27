/**
 * HttpConnectManager.java
 * com.nearme.base.http
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-8-11 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.http;

import org.apache.http.client.HttpClient;
import org.apache.http.conn.scheme.PlainSocketFactory;
import org.apache.http.conn.scheme.Scheme;
import org.apache.http.conn.scheme.SchemeRegistry;
import org.apache.http.impl.client.DefaultHttpClient;
import org.apache.http.impl.conn.tsccm.ThreadSafeClientConnManager;
import org.apache.http.params.BasicHttpParams;
import org.apache.http.params.HttpConnectionParams;
import org.apache.http.params.HttpParams;

import com.oppo.base.common.NumericUtil;

/**
 * ClassName:HttpConnectManager <br>
 * Function: http连接管理 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-8-11  上午11:59:26
 */
public class HttpConnectManager {

	private static HttpParams httpParams;  
    private static ThreadSafeClientConnManager connectionManager;  
    
    public final static String HTTP_MAX_TOTAL_CONNECTIONS = "http.max_total_connections"; 
    public final static String HTTP_MAX_ROUTE_CONNECTIONS = "http.max_route_connections"; 
    public final static String HTTP_CONNECT_TIMEOUT = "http.connect_timeout";
    public final static String HTTP_READ_TIMEOUT = "http.read_timeout";
    
    /** 
     * 最大连接数 
     */  
    public final static int MAX_TOTAL_CONNECTIONS = 1024;  
  
    /** 
     * 每个路由最大连接数 
     */  
    public final static int MAX_ROUTE_CONNECTIONS = 512;  
    /** 
     * 连接超时时间 
     */  
    public final static int CONNECT_TIMEOUT = 10000;  
    /** 
     * 读取超时时间 
     */  
    public final static int READ_TIMEOUT = 10000;  
  
    static {
    	SchemeRegistry schemeRegistry = new SchemeRegistry();  
        schemeRegistry.register(new Scheme("http", 80, PlainSocketFactory.getSocketFactory()));  
  
        //设置连接管理器
        connectionManager = new ThreadSafeClientConnManager(schemeRegistry);  
        int maxTotal = NumericUtil.parseInt(System.getProperty(HTTP_MAX_TOTAL_CONNECTIONS), MAX_TOTAL_CONNECTIONS);
        connectionManager.setMaxTotal(maxTotal);
        int maxRoute = NumericUtil.parseInt(System.getProperty(HTTP_MAX_ROUTE_CONNECTIONS), MAX_ROUTE_CONNECTIONS);
        connectionManager.setDefaultMaxPerRoute(maxRoute);
        
        httpParams = new BasicHttpParams();  
        // 设置连接超时时间  
        int connectTimeout = NumericUtil.parseInt(System.getProperty(HTTP_CONNECT_TIMEOUT), CONNECT_TIMEOUT);
        HttpConnectionParams.setConnectionTimeout(httpParams, connectTimeout);  
        // 设置读取超时时间  
        int readTimeout = NumericUtil.parseInt(System.getProperty(HTTP_READ_TIMEOUT), READ_TIMEOUT);
        HttpConnectionParams.setSoTimeout(httpParams, readTimeout);  
    }  
  
    /**
     * 获取客户端连接
     */
    public static HttpClient getHttpClient() {  
        return new DefaultHttpClient(connectionManager, httpParams);  
    }  
    
    public static void main(String[] args) {
    	HttpConnectManager.getHttpClient();
    }
}

