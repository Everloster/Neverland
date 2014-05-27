/**
 * ThemeParser.java
 * com.oppo.base.theme
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-7-28 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.theme;

import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipInputStream;

import com.oppo.base.common.StringUtil;
import com.oppo.base.file.FileOperate;
import com.oppo.base.image.ImageScaler;
import com.oppo.base.image.ImageUtil;
import com.oppo.base.xml.XmlSimpleMapParse;

/**
 * ClassName:ThemeParser
 * Function: OPPO智能机主题包解析
 * 
 * 使用方法：
 * ThemeParser tp = new ThemeParser("G:\\apk包\\主题测试包\\啤酒风格(X903).theme", 
 * 				"G:\\apk包\\主题测试包\\0.jpg", 
 * 				"G:\\apk包\\主题测试包\\1.jpg", 
 * 				"G:\\apk包\\主题测试包\\2.jpg", 
 * 				"G:\\apk包\\主题测试包\\3.jpg");
 * try {
 * 		tp.parse();
 * 		System.out.println(tp.getThemeInfo().getThemeDescription());
 * } catch(Exception ex) {
 * 		ex.printStackTrace();
 * }
 *
 * @author   80036381
 * @version  
 * @see com.oppo.base.theme.ThemeInfo
 * @since    Ver 1.1
 * @Date	 2011-7-28  下午03:08:29
 */
public class ThemeParser {
	/**
	 * 图标截图
	 */
    private static final String ICON_PATH = "assets/picture/thumbnail.jpg";
    
    /**
	 * 图标截图
	 */
    private static final String ICON_PATH_1 = "assets/picture/thumbnail.png";

    /**
     * 桌面截图
     */
    private static final String DESKTOP_PATH = "assets/picture/desktop.jpg";

    /**
     * 列表截图
     */
    private static final String LIST_MENU_PATH = "assets/picture/list.jpg";

    /**
     * 菜单截图
     */
    private static final String MAIN_MENU_PATH = "assets/picture/menu.jpg";
    
    //新主题相关文件,用户解析目前的find5的专用主题-阳波20130111
    private static final String PREVIEW_Pic_PREFIX = "preview/";
    private static final String NEW_XML_PATH = "description.xml";

    /**
     * 主题信息文件路径
     */
    private static final String XML_PATH = "assets/xml/themeinfo.xml";
    
    /**
     * 总共需要解析的文件个数，当解析个数超过此值的时候解析停止
     */
    private static final int PARSE_COUNT = 5;
    private static final int READ_BUFFER_LENGTH = 4096;

    private String themePath;	//主题路径
    private String iconPath;	//图标路径
    private String pic1;
    private String pic2;
    private String pic3;
    private ThemeInfo themeInfo;
    
    private int iconWidth;
    private int iconHeight;
    private int picWidth;
    private int picHeight;
    private float picQuality;	//截图压缩率
    private float iconQuality;	//图标压缩率
    private String picFormat;	//截图后缀

    /**
     * @param themePath	主题路径
     * @param iconPath	图标路径
     * @param pic1	截图1
     * @param pic2	截图2
     * @param pic3	截图3
     */
    public ThemeParser(String themePath, String iconPath, String  pic1, String pic2, String pic3) {
        this.themePath = themePath;
        this.iconPath = iconPath;
        this.pic1 = pic1;
        this.pic2 = pic2;
        this.pic3 = pic3;
        this.iconWidth = 72;
        this.iconHeight = 72;
        this.picWidth = 480;
        this.picHeight = 800;
        this.iconQuality = .6f;
        this.picQuality = .6f;
        this.picFormat = ImageUtil.JPG;
    }
    
    private boolean isNewThemePic(String entryName){
    	if(!StringUtil.isNullOrEmpty(entryName) && entryName.startsWith(PREVIEW_Pic_PREFIX)){
    		String suffix = entryName.substring(entryName.lastIndexOf(".") + 1);
    		if(ImageUtil.JPG.equalsIgnoreCase(suffix) || ImageUtil.JPEG.equalsIgnoreCase(suffix) || ImageUtil.PNG.equalsIgnoreCase(suffix)){
    			return true;
    		}
    	}
    	return false;
    }
    
    private String getPicPathByIndex(int index){
    	switch(index){
    	case 1: return pic1;
    	case 2: return pic2;
    	case 3: return pic3;
    	default: return null;
    	}
    }

    /**
     * 解析主题文件,得到截图及路径
     * @return 解析结果代码
     */
    public boolean parse() throws Exception {
    	themeInfo = new ThemeInfo();
    	File file = new File(themePath);
        if (!file.exists()) {
            return false;
        }
        
        ZipInputStream zipIn = null;
		InputStream zis = null;
		String iconPathEntryName = "";
		try {
			zipIn = new ZipInputStream(new BufferedInputStream(new FileInputStream(file)));	
			zis = new DataInputStream(zipIn);

			//初始化解析文件的个数
            int parseCount = 0;
            int picIndex = 1;
            ZipEntry ze;
			while(null != (ze = zipIn.getNextEntry())) {
				if(!ze.isDirectory()) {
					String name = ze.getName().toLowerCase();

					//根据文件名解析
                    if(name.equals(ICON_PATH) || name.equals(ICON_PATH_1)) {
                    	//保存截图到文件
                        copyStreamToFile(zis, iconPath, false);
                        ++parseCount;
                    } else if(name.equals(iconPathEntryName)){
                    	copyStreamToFile(zis, iconPath, false);
                		++parseCount;
                    } else if(name.equals(DESKTOP_PATH)) {
                    	//保存桌面截图到文件
                        copyStreamToFile(zis, pic1, true);
                        ++parseCount;
                    } else if(name.equals(MAIN_MENU_PATH)) {
                    	//保存菜单截图到文件
                        copyStreamToFile(zis, pic2, true);
                        ++parseCount;
                    } else if(name.equals(LIST_MENU_PATH)) {
                    	//保存列表截图到文件
                        copyStreamToFile(zis, pic3, true);
                        ++parseCount;
                    } else if(name.equals(XML_PATH)) {
                    	//解析主题信息文件获取主题信息
                    	getInfoFromStreamNew(zis);
                    	++parseCount;
                    } else if(name.equals(NEW_XML_PATH)) {
                    	Map<String, String> themeConfig = getInfoFromStreamNew1(zis);
                    	++parseCount;
                    	if(!StringUtil.isNullOrEmpty(themeConfig.get("icon"))){
                    		iconPathEntryName = themeConfig.get("icon");
                    	}
                    } else if(isNewThemePic(name)){
                    	//三张图片
                    	if(picIndex <= 3){
                    		copyStreamToFile(zis, getPicPathByIndex(picIndex), true);
                    		picIndex++;
                    		++parseCount;
                    	}
                    }
				}

				zipIn.closeEntry();
				
				//达到解析数后推出
                if (parseCount == PARSE_COUNT) {
                    break;
                }
			}
		} finally {
			FileOperate.close(zis);
			FileOperate.close(zipIn);
		}
		if(!StringUtil.isNullOrEmpty(iconPathEntryName)){
			File icon = new File(iconPath);
			if(!icon.exists()){
				ZipFile theme = new ZipFile(file);
				try{
					BufferedInputStream bis = new BufferedInputStream(theme.getInputStream(theme.getEntry(iconPathEntryName)));
					copyStreamToFile(bis, iconPath, false);
				}finally{
					//关闭此 ZipFile将关闭以前调用 getInputStream 方法返回的所有输入流
					theme.close();
				}
			}
		}
		return true;
    }

    /**
     * 将流中的数据写入到文件中
     * @param inputStream 图片流
     * @param filePath 图片保存路径
     * @param isPicture 是否是截图
     */
    private void copyStreamToFile(InputStream inputStream, String filePath, boolean isPicture) throws IOException {
    	FileOperate.createDir(filePath, true);
    	//小于等于0则不压缩
    	if((isPicture && picQuality <= 0) || (!isPicture && iconQuality <= 0)) {
        	FileOperate.saveStreamToFile(filePath, inputStream);
        	return;
    	}
    	
    	ImageScaler imgScaler = null;
        if (isPicture) {
            //按照截图压缩
        	imgScaler = new ImageScaler(inputStream, picWidth, picHeight, picFormat);
        	imgScaler.setCompressionQuality(picQuality);
        } else {
            //按照图标压缩
        	imgScaler = new ImageScaler(inputStream, iconWidth, iconHeight);
        	imgScaler.setCompressionQuality(iconQuality);
        }
        
        imgScaler.resizeWithGraphics(filePath);
    }
    
    //文件中获取主题信息
    private void getInfoFromStreamNew(InputStream inputStream) throws Exception {
    	XmlSimpleMapParse xsmp = new XmlSimpleMapParse();
    	xsmp.readFromFile(copyInputStream(inputStream));
    	
    	Map<String, String> themeConfig = xsmp.getXmlMap();
    	themeInfo.setModel(themeConfig.get("Model"));
    	themeInfo.setVersionName(themeConfig.get("Version"));
    	themeInfo.setAuthor(themeConfig.get("Author"));
    	themeInfo.setDescription(themeConfig.get("Summary"));
    	themeInfo.setEditorVersion(themeConfig.get("EditorVersion"));
    	themeInfo.setResolution(themeConfig.get("Resolution"));
    }

    //文件中获取主题信息
    private Map<String, String> getInfoFromStreamNew1(InputStream inputStream) throws Exception {
    	//未确定的几个参数先写死
    	XmlSimpleMapParse xsmp = new XmlSimpleMapParse();
    	xsmp.readFromFile(copyInputStream(inputStream));
    	
    	Map<String, String> themeConfig = xsmp.getXmlMap();
    	themeInfo.setModel("X909");
    	themeInfo.setVersionName(themeConfig.get("version"));
    	themeInfo.setAuthor(themeConfig.get("author"));
    	themeInfo.setDescription(themeConfig.get("description"));
    	themeInfo.setEditorVersion("3.0.1.0");
    	themeInfo.setResolution("1920x1080");
    	themeInfo.setPackageName(themeConfig.get("name"));
    	return themeConfig;
    }
    
    /*
    //文件中获取主题信息
    private void getInfoFromStream(InputStream inputStream) throws Exception {
    	DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
		DocumentBuilder db = dbf.newDocumentBuilder();
		//防止流被关闭，复制数据到新的流中
		Document doc = db.parse(copyInputStream(inputStream));
		
		Element rootNode = doc.getDocumentElement();
		
		//设备相关信息
		NodeList nList = rootNode.getElementsByTagName("deviceinfo");
		if(null != nList && nList.getLength() > 0 ) {
			Node deviceNode = nList.item(0);
			themeInfo.setModel(getValueFromNode(deviceNode, "model_ID"));
			themeInfo.setVersionName(getValueFromNode(deviceNode, "firmwareVersion"));
			themeInfo.setWidth(NumericUtil.parseInt(getValueFromNode(deviceNode, "lcdwidth"), 0));
			themeInfo.setHeight(NumericUtil.parseInt(getValueFromNode(deviceNode, "lcdht"), 0));
		}
		
		//主题信息
		nList = rootNode.getElementsByTagName("devicetheme");
		if(null != nList && nList.getLength() > 0 ) {
			Node themeNode = nList.item(0);
			themeInfo.setName(getValueFromNode(themeNode, "name"));
			themeInfo.setAuthor(getValueFromNode(themeNode, "author"));
			themeInfo.setDescription(getValueFromNode(themeNode, "summary"));
	        if(StringUtil.isNullOrEmpty(themeInfo.getVersionName())) {
	        	themeInfo.setVersionName(getValueFromNode(themeNode, "firmwareVersion"));
	        }
		}
    }
    
    //获取节点的指定属性值
    private String getValueFromNode(Node node, String attr) {
        try {
            Node attrNode = node.getAttributes().getNamedItem(attr);
            if(null != attrNode) {
            	return attrNode.getNodeValue();
            }
        } catch(Exception ex) {
        }
        
        return OConstants.EMPTY_STRING;
    }
    */
    
    private InputStream copyInputStream(InputStream is) {
    	ByteArrayOutputStream baos = null;
    	try {
    		baos = new ByteArrayOutputStream();
    		byte[] data = new byte[READ_BUFFER_LENGTH];
    		int readLen;
    		while((readLen = is.read(data)) != -1) {
    			baos.write(data, 0, readLen);
    		}
    		
    		return new ByteArrayInputStream(baos.toByteArray());
    	} catch(Exception ex) {
    		return null;
    	}
    }

	/**
	 * 获取主题
	 * @return  the themeInfo
	 * @since   Ver 1.0
	 */
	public ThemeInfo getThemeInfo() {
		return themeInfo;
	}
	
	/**
	 * 设置主题相关图片的压缩率
	 * @param picQuality 截图压缩率
	 * @param iconQuality 图标压缩率
	 * @return
	 */
	public void setQuality(float picQuality, float iconQuality) {
		this.picQuality = picQuality;
		this.iconQuality = iconQuality;
	}

	/**
	 * 获取iconWidth
	 * @return  the iconWidth
	 * @since   Ver 1.0
	 */
	public int getIconWidth() {
		return iconWidth;
	}

	/**
	 * 设置iconWidth
	 * @param   iconWidth    
	 * @since   Ver 1.0
	 */
	public void setIconWidth(int iconWidth) {
		this.iconWidth = iconWidth;
	}

	/**
	 * 获取iconHeight
	 * @return  the iconHeight
	 * @since   Ver 1.0
	 */
	public int getIconHeight() {
		return iconHeight;
	}

	/**
	 * 设置iconHeight
	 * @param   iconHeight    
	 * @since   Ver 1.0
	 */
	public void setIconHeight(int iconHeight) {
		this.iconHeight = iconHeight;
	}

	/**
	 * 获取picWidth
	 * @return  the picWidth
	 * @since   Ver 1.0
	 */
	public int getPicWidth() {
		return picWidth;
	}

	/**
	 * 设置picWidth
	 * @param   picWidth    
	 * @since   Ver 1.0
	 */
	public void setPicWidth(int picWidth) {
		this.picWidth = picWidth;
	}

	/**
	 * 获取picHeight
	 * @return  the picHeight
	 * @since   Ver 1.0
	 */
	public int getPicHeight() {
		return picHeight;
	}

	/**
	 * 设置picHeight
	 * @param   picHeight    
	 * @since   Ver 1.0
	 */
	public void setPicHeight(int picHeight) {
		this.picHeight = picHeight;
	}

	/**
	 * 获取截图压缩率,如果小于等于0则不压缩
	 * @return  the picQuality
	 * @since   Ver 1.0
	 */
	public float getPicQuality() {
		return picQuality;
	}

	/**
	 * 设置截图压缩率,如果小于等于0则不压缩
	 * @param   picQuality    
	 * @since   Ver 1.0
	 */
	public void setPicQuality(float picQuality) {
		this.picQuality = picQuality;
	}

	/**
	 * 获取图标压缩率,如果小于等于0则不压缩
	 * @return  the iconQuality
	 * @since   Ver 1.0
	 */
	public float getIconQuality() {
		return iconQuality;
	}

	/**
	 * 设置图标压缩率,如果小于等于0则不压缩
	 * @param   iconQuality    
	 * @since   Ver 1.0
	 */
	public void setIconQuality(float iconQuality) {
		this.iconQuality = iconQuality;
	}
	
	/**
	 * 获取截图格式
	 * @return  the picFormat
	 * @since   Ver 1.0
	 */
	public String getPicFormat() {
		return picFormat;
	}

	/**
	 * 设置截图格式
	 * @param   picFormat    
	 * @since   Ver 1.0
	 */
	public void setPicFormat(String picFormat) {
		this.picFormat = picFormat;
	}

	public static void main(String[] args) {
		//F:\\App_Store\\女性市场\\主题\\新主题\\Watermelon(R807).theme
		ThemeParser tp = new ThemeParser("theme/a.theme", 
				"theme/icon.jpg", 
				"theme/1_1.jpg", 
				"theme/2_1.jpg", 
				"theme/3_1.jpg");
		tp.setPicQuality(-1.0f);
		try {
			tp.parse();
			System.out.println(tp.getThemeInfo().getThemeDescription());
		} catch(Exception ex) {
			ex.printStackTrace();
		}
	}
}

