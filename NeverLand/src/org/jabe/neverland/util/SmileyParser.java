/*
 * Copyright (C) 2009 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.jabe.neverland.util;

import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.jabe.neverland.R;

import android.content.Context;
import android.content.res.Resources;
import android.graphics.drawable.Drawable;
import android.text.Html;
import android.text.Html.ImageGetter;
import android.text.Spannable;
import android.text.SpannableStringBuilder;
import android.text.Spanned;
import android.text.style.ImageSpan;

/**
 * A class for annotating a CharSequence with spans to convert textual emoticons
 * to graphical ones.
 */
public class SmileyParser {
	// Singleton stuff
	private volatile static SmileyParser sInstance;


	public static SmileyParser getInstance(Context context) {
		if (sInstance == null) {
			synchronized (SmileyParser.class) {
				sInstance = new SmileyParser(context);
			}
		}
		return sInstance;
	}
	
	public static SmileyParser getInstance() {
		if (sInstance == null) {
			synchronized (SmileyParser.class) {
				sInstance = new SmileyParser();
			}
		}
		return sInstance;
	}

	public static void init(Context context) {
		sInstance = new SmileyParser(context);
	}

	private SmileyParser() {
	}

	private Context mContext;
	private Resources resources;
	private Pattern mPattern;
	private ConcurrentHashMap<String, Integer> mSmileyToRes;

	private String[] mSmileyNames, mSmileyTexts;

	private SmileyParser(Context context) {
		mContext = context;
		resources = mContext.getResources();
		mSmileyNames = resources.getStringArray(DEFAULT_SMILEY_NAMES);
		mSmileyTexts = resources.getStringArray(DEFAULT_SMILEY_TEXTS);
		mSmileyToRes = buildSmileyToRes();
		mPattern = buildPattern();
	}

	static class Smileys {
		private static final int[] sIconIds = { R.drawable.bs, R.drawable.cx,
				R.drawable.dk, R.drawable.dy, R.drawable.hb, R.drawable.hx,
				R.drawable.jq, R.drawable.jw, R.drawable.k, R.drawable.kf,
				R.drawable.mg, R.drawable.mn, R.drawable.sm, R.drawable.tk,
				R.drawable.tp, R.drawable.w, R.drawable.wl, R.drawable.wx,
				R.drawable.yh, R.drawable.yj
		// , R.drawable.e101, R.drawable.e102, R.drawable.e103, R.drawable.e104,
		// R.drawable.e105,
		// R.drawable.e106, R.drawable.e107, R.drawable.e108, R.drawable.e109,
		// R.drawable.e110, R.drawable.e111,
		// R.drawable.e112, R.drawable.e113, R.drawable.e114, R.drawable.e115,
		// R.drawable.e116, R.drawable.e117,
		// R.drawable.e118, R.drawable.e119, R.drawable.e120, R.drawable.e121,
		// R.drawable.e122, R.drawable.e123,
		// R.drawable.e124, R.drawable.e125, R.drawable.e126, R.drawable.e127,
		// R.drawable.e128, R.drawable.e129,
		// R.drawable.e130, R.drawable.e131, R.drawable.e132, R.drawable.e133,
		// R.drawable.e134, R.drawable.e135,
		// R.drawable.e136, R.drawable.e137, R.drawable.e138, R.drawable.e139,
		// R.drawable.e140
		};

		public static int HAPPY = 0;
		public static int SAD = 1;
		public static int WINKING = 2;
		public static int TONGUE_STICKING_OUT = 3;
		public static int SURPRISED = 4;
		public static int KISSING = 5;
		public static int LIPS_ARE_SEALED = 6;
		public static int EMBARRASSED = 7;
		public static int CRYING = 8;
		public static int LAUGHING = 9;
		public static int UNDECIDED = 10;
		public static int MONEY_MOUTH = 11;
		public static int SLEEPY = 12;
		public static int GRIN = 13;// 坏笑
		public static int LEWD = 14;
		public static int HARPER = 15;// 木讷
		public static int SHY = 16;
		public static int SMEIL = 17;
		public static int BORINING = 18;
		public static int ANGRY = 19;

		// public static int YELLING = 6;
		// public static int COOL = 7;
		// public static int FOOT_IN_MOUTH = 9;
		// public static int ANGEL = 11;
		// public static int WTF = 16;
		// public static int EMOJ1 = ANGRY + 1;
		// public static int EMOJ2 = EMOJ1 + 1;
		// public static int EMOJ3 = EMOJ2 + 1;
		// public static int EMOJ4 = EMOJ3 + 1;
		// public static int EMOJ5 = EMOJ4 + 1;
		// public static int EMOJ6 = EMOJ5 + 1;
		// public static int EMOJ7 = EMOJ6 + 1;
		// public static int EMOJ8 = EMOJ7 + 1;
		// public static int EMOJ9 = EMOJ8 + 1;
		// public static int EMOJ10 = EMOJ9 + 1;
		// public static int EMOJ11 = EMOJ10 + 1;
		// public static int EMOJ12 = EMOJ11 + 1;
		// public static int EMOJ13 = EMOJ12 + 1;
		// public static int EMOJ14 = EMOJ13 + 1;
		// public static int EMOJ15 = EMOJ14 + 1;
		// public static int EMOJ16 = EMOJ15 + 1;
		// public static int EMOJ17 = EMOJ16 + 1;
		// public static int EMOJ18 = EMOJ17 + 1;
		// public static int EMOJ19 = EMOJ18 + 1;
		// public static int EMOJ20 = EMOJ19 + 1;
		// public static int EMOJ21 = EMOJ20 + 1;
		// public static int EMOJ22 = EMOJ21 + 1;
		// public static int EMOJ23 = EMOJ22 + 1;
		// public static int EMOJ24 = EMOJ23 + 1;
		// public static int EMOJ25 = EMOJ24 + 1;
		// public static int EMOJ26 = EMOJ25 + 1;
		// public static int EMOJ27 = EMOJ26 + 1;
		// public static int EMOJ28 = EMOJ27 + 1;
		// public static int EMOJ29 = EMOJ28 + 1;
		// public static int EMOJ30 = EMOJ29 + 1;
		// public static int EMOJ31 = EMOJ30 + 1;
		// public static int EMOJ32 = EMOJ31 + 1;
		// public static int EMOJ33 = EMOJ32 + 1;
		// public static int EMOJ34 = EMOJ33 + 1;
		// public static int EMOJ35 = EMOJ34 + 1;
		// public static int EMOJ36 = EMOJ35 + 1;
		// public static int EMOJ37 = EMOJ36 + 1;
		// public static int EMOJ38 = EMOJ37 + 1;
		// public static int EMOJ39 = EMOJ38 + 1;
		// public static int EMOJ40 = EMOJ39 + 1;

		public static int getSmileyResource(int which) {
			return sIconIds[which];
		}
	}

	// NOTE: if you change anything about this array, you must make the
	// corresponding change
	// to the string arrays: default_smiley_texts and default_smiley_names in
	// res/values/arrays.xml
	public static final int[] DEFAULT_SMILEY_RES_IDS = {
			Smileys.getSmileyResource(Smileys.HAPPY), // 0
			Smileys.getSmileyResource(Smileys.SAD), // 1
			Smileys.getSmileyResource(Smileys.WINKING), // 2
			Smileys.getSmileyResource(Smileys.TONGUE_STICKING_OUT), // 3
			Smileys.getSmileyResource(Smileys.SURPRISED), // 4
			Smileys.getSmileyResource(Smileys.KISSING), // 5
			Smileys.getSmileyResource(Smileys.LIPS_ARE_SEALED), // 6
			Smileys.getSmileyResource(Smileys.EMBARRASSED), // 7
			Smileys.getSmileyResource(Smileys.CRYING), // 8
			Smileys.getSmileyResource(Smileys.LAUGHING), // 9
			Smileys.getSmileyResource(Smileys.UNDECIDED), // 10
			Smileys.getSmileyResource(Smileys.MONEY_MOUTH), // 11
			Smileys.getSmileyResource(Smileys.SLEEPY), // 12
			Smileys.getSmileyResource(Smileys.GRIN), // 13
			Smileys.getSmileyResource(Smileys.LEWD), // 14
			Smileys.getSmileyResource(Smileys.HARPER), // 15
			Smileys.getSmileyResource(Smileys.SHY), // 16
			Smileys.getSmileyResource(Smileys.SMEIL), // 17
			Smileys.getSmileyResource(Smileys.BORINING), // 18
			Smileys.getSmileyResource(Smileys.ANGRY) // 19
	// ,
	// Smileys.getSmileyResource(Smileys.EMOJ1), // e101
	// Smileys.getSmileyResource(Smileys.EMOJ2), // e102
	// Smileys.getSmileyResource(Smileys.EMOJ3), // e103
	// Smileys.getSmileyResource(Smileys.EMOJ4), // e104
	// Smileys.getSmileyResource(Smileys.EMOJ5), // e105
	// Smileys.getSmileyResource(Smileys.EMOJ6), // e106
	// Smileys.getSmileyResource(Smileys.EMOJ7), // e107
	// Smileys.getSmileyResource(Smileys.EMOJ8), // e108
	// Smileys.getSmileyResource(Smileys.EMOJ9), // e109
	// Smileys.getSmileyResource(Smileys.EMOJ10), // e110
	// Smileys.getSmileyResource(Smileys.EMOJ11), // e111
	// Smileys.getSmileyResource(Smileys.EMOJ12), // e112
	// Smileys.getSmileyResource(Smileys.EMOJ13), // e113
	// Smileys.getSmileyResource(Smileys.EMOJ14), // e114
	// Smileys.getSmileyResource(Smileys.EMOJ15), // e115
	// Smileys.getSmileyResource(Smileys.EMOJ16), // e116
	// Smileys.getSmileyResource(Smileys.EMOJ17), // e117
	// Smileys.getSmileyResource(Smileys.EMOJ18), // e118
	// Smileys.getSmileyResource(Smileys.EMOJ19), // e119
	// Smileys.getSmileyResource(Smileys.EMOJ20), // e120
	// Smileys.getSmileyResource(Smileys.EMOJ21), // e121
	// Smileys.getSmileyResource(Smileys.EMOJ22), // e122
	// Smileys.getSmileyResource(Smileys.EMOJ23), // e123
	// Smileys.getSmileyResource(Smileys.EMOJ24), // e124
	// Smileys.getSmileyResource(Smileys.EMOJ25), // e125
	// Smileys.getSmileyResource(Smileys.EMOJ26), // e126
	// Smileys.getSmileyResource(Smileys.EMOJ27), // e127
	// Smileys.getSmileyResource(Smileys.EMOJ28), // e128
	// Smileys.getSmileyResource(Smileys.EMOJ29), // e129
	// Smileys.getSmileyResource(Smileys.EMOJ30), // e130
	// Smileys.getSmileyResource(Smileys.EMOJ31), // e131
	// Smileys.getSmileyResource(Smileys.EMOJ32), // e132
	// Smileys.getSmileyResource(Smileys.EMOJ33), // e133
	// Smileys.getSmileyResource(Smileys.EMOJ34), // e134
	// Smileys.getSmileyResource(Smileys.EMOJ35), // e135
	// Smileys.getSmileyResource(Smileys.EMOJ36), // e136
	// Smileys.getSmileyResource(Smileys.EMOJ37), // e137
	// Smileys.getSmileyResource(Smileys.EMOJ38), // e138
	// Smileys.getSmileyResource(Smileys.EMOJ39), // e139
	// Smileys.getSmileyResource(Smileys.EMOJ40), // e140
	};

	public static final int DEFAULT_SMILEY_NAMES = R.array.default_smiley_names;
	public static final int DEFAULT_SMILEY_TEXTS = R.array.default_smiley_texts;

	/**
	 * Builds the hashtable we use for mapping the string version of a smiley
	 * (e.g. ":-)") to a resource ID for the icon version.
	 */
	private ConcurrentHashMap<String, Integer> buildSmileyToRes() {
		// if (DEFAULT_SMILEY_RES_IDS.length != mSmileyTexts.length) {
		// // Throw an exception if someone updated DEFAULT_SMILEY_RES_IDS
		// // and failed to update arrays.xml
		// throw new IllegalStateException("Smiley resource ID/text mismatch");
		// }
		ConcurrentHashMap<String, Integer> smileyToRes = new ConcurrentHashMap<String, Integer>(
				mSmileyNames.length);
		for (int i = 0; i < mSmileyNames.length; i++) {
			smileyToRes.put(mSmileyNames[i], DEFAULT_SMILEY_RES_IDS[i]);
		}

		return smileyToRes;
	}

	/**
	 * Builds the regular expression we use to find smileys in
	 * {@link #addSmileySpans}.
	 */
	private Pattern buildPattern() {
		// Set the StringBuilder capacity with the assumption that the average
		// smiley is 3 characters long.
		StringBuilder patternString = new StringBuilder(mSmileyNames.length * 3);

		// Build a regex that looks like (:-)|:-(|...), but escaping the smilies
		// properly so they will be interpreted literally by the regex matcher.
		patternString.append('(');
		for (String s : mSmileyNames) {
			patternString.append(Pattern.quote(s));
			patternString.append('|');
		}
		// Replace the extra '|' with a ')'
		patternString.replace(patternString.length() - 1,
				patternString.length(), ")");

		return Pattern.compile(patternString.toString());
	}

	/**
	 * Adds ImageSpans to a CharSequence that replace textual emoticons such as
	 * :-) with a graphical version.
	 * 
	 * @param text
	 *            A CharSequence possibly containing emoticons
	 * @return A CharSequence annotated with ImageSpans covering any recognized
	 *         emoticons.
	 */
	public CharSequence addSmileySpans(CharSequence text) {
		SpannableStringBuilder builder = new SpannableStringBuilder(text);

		Matcher matcher = mPattern.matcher(text);
		while (matcher.find()) {
			int resId = mSmileyToRes.get(matcher.group());
			builder.setSpan(new ImageSpan(mContext, resId), matcher.start(),
					matcher.end(), Spannable.SPAN_EXCLUSIVE_EXCLUSIVE);
		}

		return builder;
	}

	/**
	 * 根据表情名字取资源名字
	 * 
	 * @param text
	 * @return
	 */
	public String getResourceNameSequence(String text) {
		Util.dout("getResourceNameSequence");
		for (int i = mSmileyNames.length - 1; i >= 0; i--) {
			String string = mSmileyNames[i];
			if (string.equals(text)) {
				return mSmileyTexts[i];
			}
		}
		return text;
	}

	public static Spanned getImageTextSpanString(final Context ctx,
			String content) {
		return Html.fromHtml(convertNode(content), new ImageGetter() {
			@Override
			public Drawable getDrawable(String source) {
				String packageName = ctx.getPackageName();
				try {
					Object obj;
					obj = Class.forName(packageName + ".R$drawable")
							.newInstance();
					int drawableId = Class.forName(packageName + ".R$drawable")
							.getDeclaredField(source).getInt(obj);
					if (drawableId > 0) {
						Drawable drawable = decodeDrawable(ctx,
								drawableId);
						drawable.setBounds(0, 0, drawable.getIntrinsicWidth(),
								drawable.getIntrinsicHeight());
						return drawable;
					}
				} catch (Exception e) {
					e.printStackTrace();
				}
				return null;
			}
		}, null);
	}

	/**
	 * 将[abc]转化为<img src="abc">
	 * 
	 * @param text
	 */
	private static String convertNode(String text) {
		// 提取[abc]的正则
		String xmlNodePrefix = "\\[[^\\]]+\\]";

		Pattern nodePattern = Pattern.compile(xmlNodePrefix);
		Matcher nodeMatcher = nodePattern.matcher(text);

		while (nodeMatcher.find()) {
			// 匹配到的文字
			String matchText = nodeMatcher.group();
			try {
				String resourceString = SmileyParser.getInstance()
						.getResourceNameSequence(matchText);
				String src = resourceString.substring(1,
						resourceString.length() - 1);
				String node = getNodeByValue(src);
				text = text.replace(matchText, node);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		return text;
	}

	private static String getNodeByValue(String value) {
		return "<img src='" + value + "\'>";
	}
	/**
	 * 根据resId返回Drawable
	 * @param ctx
	 * @param resId
	 * @return
	 */
	public static Drawable decodeDrawable(Context ctx, int resId){
		return ctx.getResources().getDrawable(resId);						
	}
}
