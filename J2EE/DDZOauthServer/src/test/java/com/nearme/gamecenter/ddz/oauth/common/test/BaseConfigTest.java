/**
 * Copyright (C) 2014, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	lailong
 * Since	2014年1月7日
 */
package com.nearme.gamecenter.ddz.oauth.common.test;

import org.junit.Assert;
import org.junit.Test;

import com.nearme.gamecenter.ddz.oauth.common.BaseConfig;

/**
 * 
 * @Author	LaiLong
 * @Since	2014年1月7日
 */
public class BaseConfigTest extends Assert{

	@Test
	public void testGetProjectConfig() {
		assertNotNull("failure to get port config ",BaseConfig.getProjectConfig("port"));;
		assertNotNull("failure to get deploy type config", BaseConfig.getProjectConfig("deployType"));
		assertNotNull("failure to get configure-url", BaseConfig.getProjectConfig("configure-url"));
	}
	
}
