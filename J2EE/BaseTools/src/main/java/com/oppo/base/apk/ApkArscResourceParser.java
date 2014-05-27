/**
 * ApkArscResourceParser.java
 * com.oppo.base.apk
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-7-25 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.oppo.base.apk;

import java.io.FileInputStream;
import java.io.InputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import brut.androlib.AndrolibException;
import brut.androlib.res.data.ResConfigFlags;
import brut.androlib.res.data.ResID;
import brut.androlib.res.data.ResPackage;
import brut.androlib.res.data.ResResSpec;
import brut.androlib.res.data.ResResource;
import brut.androlib.res.data.value.ResStringValue;
import brut.androlib.res.data.value.ResValue;
import brut.androlib.res.decoder.ARSCDecoder;
import brut.androlib.res.decoder.ARSCDecoder.ARSCData;

import com.oppo.base.apk.entity.ApkStringValueProperty;
import com.oppo.base.file.FileOperate;

/**
 * ClassName:ApkArscResourceParser <br>
 * Function: arsc文件解析 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-7-25  上午09:24:58
 */
public final class ApkArscResourceParser {

	//定义国际化简体中文(-zh-rCN)flag  主要是第三四个参数  其他参数都用默认的
	public static final ResConfigFlags zhCNFlag = new ResConfigFlags((short)0, (short)0, new char[]{'z', 'h'}, new char[]{'C', 'N'},
			(short)0, (byte)0, (byte)0, (short)0, (byte)0, (byte)0, (byte)0,
			(short)0, (short)0, (short)0, (byte)0, (byte)0,
			(short)0, (short)0, (short)0, false);

	public static Map<String, ApkStringValueProperty> getProperties(InputStream in) throws AndrolibException {
		//得到解析数据
		ARSCData data = ARSCDecoder.decode(in, true, true);

		Map<String, ApkStringValueProperty> propMap = new HashMap<String, ApkStringValueProperty>();
		ResPackage[] resPacks = data.getPackages();

		//遍历包
		for(ResPackage resPack : resPacks) {
			List<ResResSpec> rrsList = resPack.listResSpecs();
			for(ResResSpec rrs : rrsList) {
				ResResource rResource = getResResource(rrs, zhCNFlag);
				if(rResource != null){
					ResValue rrv = rResource.getValue();
					//目前只需要获取string类型的值
					if(rrv instanceof ResStringValue) {
						ApkStringValueProperty prop = new ApkStringValueProperty();

						ResID resId = rrs.getId();
						prop.setId(resId.id);
						prop.setIdString(resId.toString());
						prop.setName(rrs.getName());
						prop.setValue(((ResStringValue)rrv).encodeAsResXmlValue());

						propMap.put(prop.getIdString(), prop);
					}
				}
			}
		}

		return propMap;
	}
	
	private static ResResource getResResource(ResResSpec rrs, ResConfigFlags resConfigFlags) throws AndrolibException {
		ResResource rResource = null;
		try {
			rResource = rrs.getResource(resConfigFlags);
		} catch (AndrolibException e) {
		}
		
		if (null != rResource) {
			return rResource;
		}
		
		if (rrs.hasDefaultResource()) {
			return rrs.getDefaultResource();
		}
		
		return null;
	}

	public static void main(String[] args) {
		FileInputStream fis = null;
		try {
			fis = new FileInputStream("h:\\resources.arsc");
			Map<String, ApkStringValueProperty> props = ApkArscResourceParser.getProperties(fis);
			for(String key : props.keySet()) {
				System.out.println(key + "\t" + props.get(key));
			}
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			FileOperate.close(fis);
		}
	}
}

