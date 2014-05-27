/**
 * ProxoolFactory.java
 * com.oppo.base.db
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-10-7 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.oppo.base.db;

import java.util.Properties;

import javax.sql.DataSource;

import org.apache.ibatis.datasource.DataSourceFactory;
import org.logicalcobwebs.proxool.ProxoolDataSource;
import org.logicalcobwebs.proxool.ProxoolFacade;

import com.oppo.base.common.NumericUtil;

/**
 * ClassName:ProxoolFactory
 * Function: Proxool连接池管理
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-10-7  上午11:01:47
 */
public class ProxoolFactory implements DataSourceFactory {
	
	//初始化Proxool管理的数据源
	private ProxoolDataSource dataSource;
	
	/**
	 * 线程池是否关闭
	 */
	private volatile static boolean isPoolShutdown = true;
	
	public DataSource getDataSource() {
		isPoolShutdown = false;
		return dataSource;
	}

	public void setProperties(Properties prop) {
		//设置数据源属性
		dataSource = new ProxoolDataSource();
		dataSource.setDriver(prop.getProperty("driver"));
		dataSource.setDriverUrl(prop.getProperty("url"));
		dataSource.setUser(prop.getProperty("username"));
		//密码采用
		dataSource.setPassword(getPassword(prop.getProperty("password")));
		dataSource.setAlias(prop.getProperty("alias"));
		
		//设置最大连接数
		if(prop.containsKey("maximumConnectionCount")) {
			dataSource.setMaximumConnectionCount(
				NumericUtil.parseInt(prop.getProperty("maximumConnectionCount"), 25));
		}
		//设置最小连接数
		if(prop.containsKey("minimumConnectionCount")) {
			dataSource.setMinimumConnectionCount(
				NumericUtil.parseInt(prop.getProperty("minimumConnectionCount"), 5));
		}
		//设置最大线程活动时间
		if(prop.containsKey("maximumActiveTime")) {
			dataSource.setMaximumActiveTime(
				NumericUtil.parseLong(prop.getProperty("maximumActiveTime"), 5 * 60 * 1000));
		}
		//设置连接存在的最长保持活动的时间
		if(prop.containsKey("maximumConnectionLifetime")) {
			dataSource.setMaximumConnectionLifetime(
				NumericUtil.parseInt(prop.getProperty("maximumConnectionLifetime"), 5 * 60 * 1000));
		}
		//设置测试连接的SQL,及测试间隔
		if(prop.containsKey("houseKeepingTestSql")) {
			dataSource.setHouseKeepingTestSql(prop.getProperty("houseKeepingTestSql"));
			if(prop.containsKey("houseKeepingSleepTime")) {
				dataSource.setHouseKeepingSleepTime(
					NumericUtil.parseInt(prop.getProperty("houseKeepingSleepTime"), 30 * 1000));
			}
		}
		//设置同时创建的连接数
		if(prop.containsKey("simultaneousBuildThrottle")) {
			dataSource.setSimultaneousBuildThrottle(
					NumericUtil.parseInt(prop.getProperty("simultaneousBuildThrottle"), 10));
		}
		
		//是否记录执行的SQL
		if(prop.containsKey("trace")) {
			dataSource.setTrace(Boolean.parseBoolean(prop.getProperty("trace")));
		}
	}
	
	/**
	 * 根据配置的密码获取真实密码
	 * @param 
	 * @return
	 */
	protected String getPassword(String password) {
		return password;
	}
	
	/**
	 * 关闭连接池
	 * @param 
	 * @return
	 */
	public static void shutdown() {
		if(!isPoolShutdown) {
			isPoolShutdown = true;
			ProxoolFacade.shutdown();
		}
	}
}

