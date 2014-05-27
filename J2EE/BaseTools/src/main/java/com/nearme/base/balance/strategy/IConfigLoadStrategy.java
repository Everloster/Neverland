/**
 * IConfigLoadStrategy.java
 * com.nearme.base.balance.strategy
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-11-12 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.balance.strategy;

import com.nearme.base.balance.model.ServerConfig;
import com.nearme.base.balance.model.ServerGroupInfo;
import com.nearme.base.balance.parser.IConfigParser;

/**
 * ClassName:IConfigLoadStrategy <br>
 * 解析器加载数据的策略 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-11-12  上午09:49:29
 */
public interface IConfigLoadStrategy {
	/**
	 * 通过指定解析器解析出配置
	 * @param 
	 * @return
	 */
	ServerConfig parse() throws Exception;
	
	/**
	 * 根据指定对象解析指定的id成服务组配置
	 * @param 
	 * @return
	 */
	ServerGroupInfo parseWithId(String id) throws Exception;
	
	/**
	 * 获取配置解析器，利用此解析器来解析服务信息
	 * @param 
	 * @return
	 */
	IConfigParser getConfigParser();
	
	/**
	 * 配置文件是否需要重新加载
	 * @param 
	 * @return
	 */
	boolean needLoad();
}

