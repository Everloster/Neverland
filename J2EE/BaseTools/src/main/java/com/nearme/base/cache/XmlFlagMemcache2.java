/**
 * XmlFlagMemcache2.java
 * com.nearme.base.cache
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-12-13 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.nearme.base.cache;

import java.io.FileOutputStream;
import java.util.Map;

import com.oppo.base.common.NumericUtil;
import com.oppo.base.common.StringUtil;
import com.oppo.base.file.FileOperate;
import com.oppo.base.xml.XmlConfigReader;
import com.oppo.base.xml.XmlUtil;

/**
 * ClassName:XmlFlagMemcache2 <br>
 * Function: key带标记的memcache,需要调用setFlagFilePath指定flag保存的xml文件路径 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-12-13  下午08:13:22
 */
public class XmlFlagMemcache2 extends Memcache2 {
	
	private String flagFilePath;
	
	/**
	 * 将传入的cacheId转换为最终存储的cacheId,在转换后增加一个标记，
	 * @param cacheId
	 * @return
	 */
	@Override
	protected String transferCacheId(String cacheId) {
		String key = super.transferCacheId(cacheId);
		
		return key + getKeyFlag();
	}
	
	/**
	 * 获取缓存key的标记,此处为利用文件进行存储 <br>
	 * 如果需要改为其他方式存储flag,则需要重写此方法 <br>
	 * @param 
	 * @return
	 */
	protected String getKeyFlag() {
		return getKeyFlag(false);
	}
	
	/**
	 * 升级缓存key的版本
	 * @return 更新后的版本
	 */
	public String updateKeyVersion() {
		return getKeyFlag(true);
	}
	
	/**
	 * 获得key版本
	 * @param update 是否需要升级
	 * @return
	 */
	public String getKeyFlag(boolean update) {
		String prefix = this.getPrefix();
		
		//从配置文件中读取flag
		XmlConfigReader configReader = XmlConfigReader.getXmlConfig(flagFilePath);
		Map<String, String> map = configReader.getConfigMap();
		String flag = map.get(prefix);
		
		//如果未取到则默认为0，并将默认值进行存储，方便flag升级
		if(StringUtil.isNullOrEmpty(flag)) {
			//默认为0
			flag = "0";
		}
		
		if(update) {
			flag = String.valueOf(NumericUtil.parseInt(flag, -1) + 1);
			map.put(prefix, flag);
		
			//将标记存入xml中
			FileOutputStream fos = null;
			try {
				fos = new FileOutputStream(flagFilePath);
				XmlUtil.transMapToOutput(map, fos, "memcache-flag", false);
			} catch(Exception ex) {
			} finally {
				FileOperate.close(fos);
			}
		}
		
		return flag;
	}

	/**
	 * 获取标记文件的路径
	 * @return  the flagFilePath
	 * @since   Ver 1.0
	 */
	public String getFlagFilePath() {
		return flagFilePath;
	}

	/**
	 * 设置标记文件的路径
	 * @param   flagFilePath    
	 * @since   Ver 1.0
	 */
	public void setFlagFilePath(String flagFilePath) {
		this.flagFilePath = flagFilePath;
	}
	
}

