/**
 * XmlFileConfigLoadParser.java
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

import java.io.File;

import com.nearme.base.balance.model.ServerConfig;
import com.nearme.base.balance.model.ServerGroupInfo;
import com.nearme.base.balance.parser.IConfigParser;

/**
 * ClassName:XmlFileConfigLoadParser <br>
 * Function: xml文件配置解析 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-11-12  上午09:58:12
 */
public class XmlFileConfigLoadParser implements IConfigLoadStrategy {
	
	private File configFile;
	private volatile long lastLoadTime;	//最后一次加载时间
	private long lastModifyTime;	//配置文件修改时间
	private IConfigParser parser;
	
	public XmlFileConfigLoadParser(IConfigParser parser, File xmlFile) {
		this.configFile = xmlFile;
		this.parser = parser;
	}
	
	public XmlFileConfigLoadParser(IConfigParser parser, String xmlFilePath) {
		this(parser, new File(xmlFilePath));
	}

	@Override
	public ServerConfig parse() throws Exception {
		return getConfigParser().parse(configFile);
	}
	
	@Override
	public ServerGroupInfo parseWithId(String id) throws Exception {
		return getConfigParser().parseWithId(configFile, id);
	}
	
	@Override
	public IConfigParser getConfigParser() {
		return parser;
	}

	@Override
	public boolean needLoad() {
		long now = System.currentTimeMillis();
		//五分钟内不重新去读
		if(now - this.lastLoadTime < 5 * 60 * 1000) {
			return false;
		}
		this.lastLoadTime = now;
		
		//文件不存在则不重新读
		if(!configFile.exists()) {
			return false;
		}
		
		//文件未改变则不重新读
		long fileModifyTime = configFile.lastModified();
		if(lastModifyTime == fileModifyTime) {
			return false;
		}
		lastModifyTime = fileModifyTime;
		
		return true;
	}
}

