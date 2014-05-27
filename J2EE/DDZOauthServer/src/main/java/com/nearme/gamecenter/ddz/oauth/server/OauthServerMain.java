/**
 * Copyright (C) 2014, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	lailong
 * Since	2014年1月7日
 */
package com.nearme.gamecenter.ddz.oauth.server;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.nearme.gamecenter.ddz.oauth.common.BaseConfig;
import com.nearme.gamecenter.ddz.redis.RedisServer;
import com.oppo.base.common.NumericUtil;

/**
 * 
 * @Author	LaiLong
 * @Since	2014年1月7日
 */
public class OauthServerMain {

	private static final Logger logger = LoggerFactory.getLogger(OauthServerMain.class);
	
	/**
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		
		logger.info("OauthServer is ready to start.");
		
		final int port = NumericUtil.parseInt(BaseConfig.getProjectConfig("port"), NettyConstants.DEFAULT_SERVER_PORT);
		
		final ClientAccessHttpServer httpServer = new ClientAccessHttpServer();
		httpServer.start(port);
		
		RedisServer.getInstance().startServer();
	
		logger.info("ClientAccess server start on port " + port);
	}

}
