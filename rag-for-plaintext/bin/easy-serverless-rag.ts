#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { BedrockKnowledgeBaseStack } from '../lib/BedrockKnowledgeBaseStack';

const app = new cdk.App();
new BedrockKnowledgeBaseStack(app, 'BedrockKnowledgeBaseStack', {
});

app.synth();
