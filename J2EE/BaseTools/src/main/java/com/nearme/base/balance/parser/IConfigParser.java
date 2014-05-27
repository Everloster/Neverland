/**
 * IConfigParser.java
 * com.nearme.base.balance.parser
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-11-12 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.balance.parser;

import com.nearme.base.balance.model.ServerConfig;
import com.nearme.base.balance.model.ServerGroupInfo;

/**
 * ClassName:IConfigParser <br>
 * Function: 配置解析器 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-11-12  上午09:51:20
 */
public interface IConfigParser {
	/**
	 * 根据指定对象解析成服务器配置
	 * @param 
	 * @return
	 */
	ServerConfig parse(Object xmlObject) throws Exception;
	
	/**
	 * 根据指定对象解析指定的id成服务组配置
	 * @param 
	 * @return
	 */
	ServerGroupInfo parseWithId(Object xmlObject, String id) throws Exception;
}

