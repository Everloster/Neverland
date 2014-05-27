/**
 * InputStreamConfigLoadParser.java
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

import java.io.InputStream;

import com.nearme.base.balance.model.ServerConfig;
import com.nearme.base.balance.model.ServerGroupInfo;
import com.nearme.base.balance.parser.IConfigParser;

/**
 * ClassName:InputStreamConfigLoadParser <br>
 * Function: InputStream类型的配置解析 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-11-12  上午10:08:43
 */
public class InputStreamConfigLoadParser implements IConfigLoadStrategy {
	
	private InputStream inputStream;
	private IConfigParser parser;
	
	public InputStreamConfigLoadParser(IConfigParser parser, InputStream inputStream) {
		this.inputStream = inputStream;
		this.parser = parser;
	}
	
	@Override
	public ServerConfig parse() throws Exception {
		return getConfigParser().parse(inputStream);
	}
	
	/**
	 * 根据指定对象解析指定的id成服务组配置
	 * @param 
	 * @return
	 */
	@Override
	public ServerGroupInfo parseWithId(String id) throws Exception {
		return getConfigParser().parseWithId(inputStream, id);
	}
	
	/**
	 * 获取配置解析器
	 * @param 
	 * @return
	 */
	@Override
	public IConfigParser getConfigParser() {
		return parser;
	}

	@Override
	public boolean needLoad() {
		//InputStream类型不需要重新加载
		return false;
	}
}

