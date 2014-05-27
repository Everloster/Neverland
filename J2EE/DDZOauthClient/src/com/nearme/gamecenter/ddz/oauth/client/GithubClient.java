/**
 * Copyright (C) 2014, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	lailong
 * Since	2014年2月10日
 */
package com.nearme.gamecenter.ddz.oauth.client;

import java.util.List;

import retrofit.RestAdapter;
import retrofit.http.GET;
import retrofit.http.Path;

/**
 * 
 * @Author LaiLong
 * @Since 2014年2月10日
 */
public class GithubClient {

	private static final String API_URL = "https://api.github.com";

	static class Contributor {
		String login;
		int contributions;
	}

	public interface Github {
		@GET("/repos/{owner}/{repo}/contributors")
		List<Contributor> contributors(@Path("owner") String owner,
				@Path("repo") String repo);
	}
	
	public static void main(String[] args) {
		// create a simple REST adapter 
		RestAdapter restAdapter = new RestAdapter.Builder().setEndpoint(API_URL).build();
		
		// create an interface for github 
		Github githubApi = restAdapter.create(Github.class);
		
		// Fetch and print a list of the contributors to this library.
	    List<Contributor> contributors = githubApi.contributors("square", "retrofit");
	    for (Contributor contributor : contributors) {
	      System.out.println(contributor.login + " (" + contributor.contributions + ")");
	    }
	}
}
